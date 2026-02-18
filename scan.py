#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily Ticker Report (GitHub Pages)

Agenda @22:30:
1) Market recap & positioning
   - Cross-asset table (US + Europe + rates + FX + commodities + BTC)
   - Multi-horizon returns: 1D / 7D / 1M / 3M / 6M (trend read)
   - Commentary: interpretation based on cross-asset + multi-horizon + RSS themes
2) Biggest movers (>=4%): session + after-hours (gainers/losers)
4) Technical triggers:
   4A Early callouts (~80%): within 0.5 ATR of trigger
   4B Breakouts/breakdowns: SOFT (pierced but <0.5 ATR) + CONFIRMED (>=0.5 ATR)
5) Needle-moving catalysts: RSS headline digest + interpreted themes (not copy/paste)

Reliability features:
- robust yfinance chunk download (MultiIndex safe)
- movers tables never crash (always returns schema)
- session movers fallback: compute from scanned universe if scraping fails
- after-hours movers fallback: compute from custom tickers pre/post data
- headless matplotlib backend for GitHub Actions
- always writes docs/index.md and docs/report.md; if anything fails, publishes traceback and exits 0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR = BASE_DIR / "docs"
IMG_DIR = DOCS_DIR / "img"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DOCS_DIR / "state.json"
REPORT_PATH = DOCS_DIR / "report.md"
INDEX_PATH = DOCS_DIR / "index.md"
NOJEKYLL_PATH = DOCS_DIR / ".nojekyll"

CUSTOM_TICKERS_PATH = CONFIG_DIR / "tickers_custom.txt"
SP500_LOCAL = CONFIG_DIR / "universe_sp500.txt"
NDX_LOCAL = CONFIG_DIR / "universe_nasdaq100.txt"


# ----------------------------
# Config knobs
# ----------------------------
MOVER_THRESHOLD_PCT = 4.0

ATR_N = 14
ATR_CONFIRM_MULT = 0.5
EARLY_MULT = 0.5

LOOKBACK_DAYS = 260 * 2
DOWNLOAD_PERIOD = "3y"
DOWNLOAD_INTERVAL = "1d"
CHUNK_SIZE = 80

MAX_CHARTS_EARLY = 14
MAX_CHARTS_TRIGGERED = 18

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


# ----------------------------
# Helpers: IO
# ----------------------------
def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict) -> None:
    write_text(STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False))


# ----------------------------
# Web fetch (HTML/RSS) stdlib only
# ----------------------------
def fetch_url_text(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def read_html_tables(url: str) -> List[pd.DataFrame]:
    html = fetch_url_text(url)
    return pd.read_html(html)


def parse_rss(url: str, source_name: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Minimal RSS parser:
    Returns list of dict(title, link, pubDate, source)
    """
    try:
        xml_text = fetch_url_text(url, timeout=30)
        root = ET.fromstring(xml_text)

        # RSS: channel/item, Atom: entry
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                items.append({"title": title, "link": link, "pubDate": pub, "source": source_name})
        if items:
            return items[:limit]

        # Atom fallback
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link", ns)
            link = (link_el.get("href") if link_el is not None else "").strip()
            pub = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
            if title:
                items.append({"title": title, "link": link, "pubDate": pub, "source": source_name})
        return items[:limit]
    except Exception:
        return []


def fetch_rss_headlines(limit_total: int = 14) -> List[Dict[str, str]]:
    """
    RSS sources:
    - Yahoo Finance Top Stories RSS (often works even when yfinance.news fails)
    - CNBC Top News + CNBC Markets
    """
    feeds = [
        ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
        ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("CNBC Markets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
    ]
    all_items: List[Dict[str, str]] = []
    for name, url in feeds:
        all_items.extend(parse_rss(url, name, limit=12))

    # De-dupe by title
    seen = set()
    uniq = []
    for it in all_items:
        t = it.get("title", "")
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(it)

    return uniq[:limit_total]


# ----------------------------
# Universe
# ----------------------------
def _clean_ticker(t: str) -> str:
    t = str(t).strip()
    # Wikipedia uses BRK.B -> Yahoo uses BRK-B
    if re.fullmatch(r"[A-Z]+\.[A-Z]+", t):
        return t.replace(".", "-")
    return t


def get_sp500_tickers() -> List[str]:
    local = read_lines(SP500_LOCAL)
    if local:
        return sorted({_clean_ticker(x) for x in local})

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = [_clean_ticker(x) for x in df["Symbol"].astype(str).tolist()]
        return sorted(set(tickers))
    except Exception:
        return []


def get_nasdaq100_tickers() -> List[str]:
    local = read_lines(NDX_LOCAL)
    if local:
        return sorted({_clean_ticker(x) for x in local})

    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)

        df = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "ticker" in cols or "symbol" in cols or any("ticker" in str(c).lower() for c in t.columns):
                df = t
                break
        if df is None:
            return []

        col = None
        for c in df.columns:
            if str(c).lower() in ("ticker", "symbol"):
                col = c
                break
        if col is None:
            col = df.columns[0]

        tickers = [_clean_ticker(x) for x in df[col].astype(str).tolist()]
        tickers = [t for t in tickers if re.fullmatch(r"[\w\-\.\=]+", t)]
        return sorted(set(tickers))
    except Exception:
        return []


def get_custom_tickers() -> List[str]:
    return sorted({_clean_ticker(x) for x in read_lines(CUSTOM_TICKERS_PATH)})


# ----------------------------
# Market data (yfinance) - robust extraction
# ----------------------------
def extract_ohlcv_from_download(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None

    # Single ticker: flat columns
    if not isinstance(data.columns, pd.MultiIndex):
        if not set(["Open", "High", "Low", "Close"]).issubset(set(data.columns)):
            return None
        df = data.copy()
        df.index.name = df.index.name or "Date"
        keep = [c for c in FIELDS if c in df.columns]
        df = df[keep].dropna(subset=["Close"])
        return df if not df.empty else None

    cols = data.columns

    # Orientation A: (Field, Ticker)
    if ("Close", ticker) in cols:
        df = pd.DataFrame({
            "Open": data[("Open", ticker)] if ("Open", ticker) in cols else np.nan,
            "High": data[("High", ticker)] if ("High", ticker) in cols else np.nan,
            "Low": data[("Low", ticker)] if ("Low", ticker) in cols else np.nan,
            "Close": data[("Close", ticker)],
            "Volume": data[("Volume", ticker)] if ("Volume", ticker) in cols else np.nan,
        })
        df.index.name = df.index.name or "Date"
        df = df.dropna(subset=["Close"])
        return df if not df.empty else None

    # Orientation B: (Ticker, Field)
    if (ticker, "Close") in cols:
        df = pd.DataFrame({
            "Open": data[(ticker, "Open")] if (ticker, "Open") in cols else np.nan,
            "High": data[(ticker, "High")] if (ticker, "High") in cols else np.nan,
            "Low": data[(ticker, "Low")] if (ticker, "Low") in cols else np.nan,
            "Close": data[(ticker, "Close")],
            "Volume": data[(ticker, "Volume")] if (ticker, "Volume") in cols else np.nan,
        })
        df.index.name = df.index.name or "Date"
        df = df.dropna(subset=["Close"])
        return df if not df.empty else None

    return None


def yf_download_chunk(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        if not chunk:
            continue

        try:
            data = yf.download(
                tickers=chunk,
                period=DOWNLOAD_PERIOD,
                interval=DOWNLOAD_INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if data is None or data.empty:
            continue

        for t in chunk:
            df = extract_ohlcv_from_download(data, t)
            if df is not None and not df.empty:
                out[t] = df

    return out


def pct_change_last(df: pd.DataFrame) -> Optional[float]:
    c = df["Close"].dropna()
    if len(c) < 2:
        return None
    return float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ----------------------------
# Snapshot: cross-asset + multi-horizon
# ----------------------------
def _extract_close_series(download_df: pd.DataFrame, sym: str) -> Optional[pd.Series]:
    if download_df is None or download_df.empty:
        return None
    if not isinstance(download_df.columns, pd.MultiIndex):
        if "Close" in download_df.columns:
            return download_df["Close"].dropna()
        return None
    cols = download_df.columns
    if ("Close", sym) in cols:
        return download_df[("Close", sym)].dropna()
    if (sym, "Close") in cols:
        return download_df[(sym, "Close")].dropna()
    return None


def _color_pct_cell(x: float) -> str:
    # emoji always visible; HTML span often works on Pages
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if x > 0:
        return f'🟩 <span style="color:#11823b;">{x:+.2f}%</span>'
    if x < 0:
        return f'🟥 <span style="color:#b91c1c;">{x:+.2f}%</span>'
    return f'⬜ <span style="color:#6b7280;">{x:+.2f}%</span>'


def _one_day_return(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    if prev == 0:
        return float("nan")
    return (last / prev - 1.0) * 100.0


def _return_since(series: pd.Series, days_back: int) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")

    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    s2 = s.copy()
    s2.index = idx

    last_dt = pd.Timestamp(s2.index[-1])
    target = last_dt - pd.Timedelta(days=days_back)

    past = s2.loc[:target]
    if past.empty:
        return float("nan")

    last = float(s2.iloc[-1])
    base = float(past.iloc[-1])
    if base == 0:
        return float("nan")
    return (last / base - 1.0) * 100.0


def fetch_market_snapshot_multi() -> pd.DataFrame:
    """
    Instruments requested (plus core US tape & risk proxies).
    Europe:
      - STOXX Europe 600: ^STOXX
      - DAX: ^GDAXI
      - CAC 40: ^FCHI
      - FTSE 100: ^FTSE
    Commodities:
      - Gold: GC=F, Silver: SI=F, Coffee: KC=F, Cocoa: CC=F
    FX/Crypto:
      - EURUSD=X, BTC-USD
    """
    instruments = [
        ("Nasdaq 100", "^NDX"),
        ("S&P 500", "^GSPC"),

        ("STOXX Europe 600", "^STOXX"),
        ("DAX", "^GDAXI"),
        ("CAC 40", "^FCHI"),
        ("FTSE 100", "^FTSE"),

        ("VIX", "^VIX"),
        ("US 10Y", "^TNX"),
        ("DXY", "DX-Y.NYB"),
        ("EUR/USD", "EURUSD=X"),

        ("WTI", "CL=F"),
        ("Gold", "GC=F"),
        ("Silver", "SI=F"),
        ("Coffee", "KC=F"),
        ("Cocoa", "CC=F"),

        ("Bitcoin", "BTC-USD"),
    ]

    syms = [s for _, s in instruments]

    try:
        data = yf.download(
            tickers=syms,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame(columns=["Instrument", "Symbol", "Last", "1D", "7D", "1M", "3M", "6M"])

    rows = []
    for name, sym in instruments:
        close = _extract_close_series(data, sym)
        if close is None or close.dropna().empty:
            continue
        close = close.dropna()
        last = float(close.iloc[-1])

        rows.append({
            "Instrument": name,
            "Symbol": sym,
            "Last": last,
            "1D": _one_day_return(close),
            "7D": _return_since(close, 7),
            "1M": _return_since(close, 30),
            "3M": _return_since(close, 90),
            "6M": _return_since(close, 180),
        })

    return pd.DataFrame(rows)


def format_snapshot_table_multi(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_Snapshot unavailable._"

    d = df.copy()
    d["Last"] = pd.to_numeric(d["Last"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    for c in ["1D", "7D", "1M", "3M", "6M"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").map(_color_pct_cell)

    cols = ["Instrument", "Last", "1D", "7D", "1M", "3M", "6M"]
    return d[cols].to_markdown(index=False)


# ----------------------------
# Commentary: interpret, not copy/paste
# ----------------------------
def summarize_rss_themes(items: List[Dict[str, str]]) -> str:
    """
    Converts headlines into a theme summary (not a copy-paste of headlines).
    """
    if not items:
        return "No RSS headlines were available at runtime."

    text = " ".join([it.get("title", "") for it in items]).lower()

    themes = []
    if any(k in text for k in ["fed", "fomc", "minutes", "powell", "rates", "yield", "treasury", "inflation"]):
        themes.append("Rates/Fed narrative was prominent (policy path / yields / inflation).")
    if any(k in text for k in ["ai", "chip", "semiconductor", "nvidia", "software", "cloud"]):
        themes.append("AI/tech positioning remained a key driver (chips vs software sensitivity).")
    if any(k in text for k in ["earnings", "guidance", "forecast", "results"]):
        themes.append("Earnings/guidance headlines likely drove single-name dispersion.")
    if any(k in text for k in ["oil", "crude", "wti", "energy", "opec"]):
        themes.append("Energy/oil was a factor (inflation impulse / geopolitics).")
    if any(k in text for k in ["china", "tariff", "europe", "ecb", "boj", "war", "geopolitic"]):
        themes.append("Macro/geopolitics featured in risk appetite and sector rotation.")

    if not themes:
        themes = ["Headline mix looked broad; no single macro theme dominated the feed."]

    return " ".join(themes)


def build_commentary(snapshot_df: pd.DataFrame, rss_items: List[Dict[str, str]]) -> str:
    """
    Cross-asset + multi-horizon interpretation.
    Goal: explain the day AND the wider context (7D/1M/3M/6M).
    """
    if snapshot_df is None or snapshot_df.empty:
        return "Commentary unavailable (snapshot table empty)."

    def row(name: str) -> Optional[pd.Series]:
        x = snapshot_df.loc[snapshot_df["Instrument"] == name]
        return None if x.empty else x.iloc[0]

    ndx = row("Nasdaq 100")
    spx = row("S&P 500")
    stoxx = row("STOXX Europe 600")
    dax = row("DAX")
    vix = row("VIX")
    tnx = row("US 10Y")
    dxy = row("DXY")
    eurusd = row("EUR/USD")
    wti = row("WTI")
    gold = row("Gold")
    btc = row("Bitcoin")

    bullets: List[str] = []

    # --- Day regime: risk-on/off ---
    if spx is not None and vix is not None and pd.notna(spx["1D"]) and pd.notna(vix["1D"]):
        if spx["1D"] > 0 and vix["1D"] < 0:
            bullets.append(
                f"**Risk-on session**: equities higher while volatility fell (S&P {spx['1D']:+.2f}%, VIX {vix['1D']:+.2f}%)."
            )
        elif spx["1D"] < 0 and vix["1D"] > 0:
            bullets.append(
                f"**Risk-off session**: equities lower while volatility rose (S&P {spx['1D']:+.2f}%, VIX {vix['1D']:+.2f}%)."
            )
        else:
            bullets.append(
                f"Mixed risk tone (S&P {spx['1D']:+.2f}%, VIX {vix['1D']:+.2f}%)."
            )

    # --- Leadership / rotation ---
    if ndx is not None and spx is not None and pd.notna(ndx["1D"]) and pd.notna(spx["1D"]):
        if ndx["1D"] > spx["1D"]:
            bullets.append(f"**Tech leadership** on the day: Nasdaq outperformed (NDX {ndx['1D']:+.2f}% vs S&P {spx['1D']:+.2f}%).")
        else:
            bullets.append(f"**Broader leadership** on the day: S&P kept pace/outperformed (S&P {spx['1D']:+.2f}% vs NDX {ndx['1D']:+.2f}%).")

    # --- Europe read-through ---
    if stoxx is not None and spx is not None and pd.notna(stoxx["1D"]) and pd.notna(spx["1D"]):
        bullets.append(f"Europe vs US: STOXX 600 {stoxx['1D']:+.2f}% vs S&P {spx['1D']:+.2f}% (relative strength check).")
    if dax is not None and pd.notna(dax["1D"]):
        bullets.append(f"Germany: DAX {dax['1D']:+.2f}% (cyclical/industrial risk appetite proxy).")

    # --- Rates / FX ---
    if tnx is not None and pd.notna(tnx["1D"]):
        bullets.append(f"Rates: US 10Y moved {tnx['1D']:+.2f}% on the day (key input into duration/tech multiples).")
    if dxy is not None and eurusd is not None and pd.notna(dxy["1D"]) and pd.notna(eurusd["1D"]):
        bullets.append(f"FX: DXY {dxy['1D']:+.2f}% vs EUR/USD {eurusd['1D']:+.2f}% (USD impulse into commodities & global earnings).")

    # --- Commodities / hedges ---
    if wti is not None and pd.notna(wti["1D"]):
        bullets.append(f"Energy: WTI {wti['1D']:+.2f}% (inflation impulse / risk sentiment).")
    if gold is not None and pd.notna(gold["1D"]):
        bullets.append(f"Gold: {gold['1D']:+.2f}% (hedge bid vs real rates).")
    if btc is not None and pd.notna(btc["1D"]):
        bullets.append(f"Crypto: BTC {btc['1D']:+.2f}% (risk beta / liquidity proxy).")

    # --- Wider context: multi-horizon trend framing ---
    def trend_phrase(inst: str, r1m: float, r3m: float, r6m: float) -> str:
        if pd.isna(r1m) or pd.isna(r3m) or pd.isna(r6m):
            return f"{inst}: insufficient history for multi-horizon trend."
        if r1m > 0 and r3m > 0 and r6m > 0:
            return f"{inst} remains in an **uptrend** across 1M/3M/6M (+{r1m:.2f}% / +{r3m:.2f}% / +{r6m:.2f}%)."
        if r1m < 0 and r3m < 0 and r6m < 0:
            return f"{inst} remains in a **downtrend** across 1M/3M/6M ({r1m:.2f}% / {r3m:.2f}% / {r6m:.2f}%)."
        if r1m > 0 and r3m < 0:
            return f"{inst}: **bounce inside a pullback** (1M +{r1m:.2f}% but 3M {r3m:.2f}%)."
        if r1m < 0 and r3m > 0:
            return f"{inst}: **pullback inside a larger uptrend** (1M {r1m:.2f}% but 3M +{r3m:.2f}%)."
        return f"{inst}: mixed trend (1M {r1m:+.2f}%, 3M {r3m:+.2f}%, 6M {r6m:+.2f}%)."

    if spx is not None:
        bullets.append(trend_phrase("S&P", spx.get("1M", np.nan), spx.get("3M", np.nan), spx.get("6M", np.nan)))
    if ndx is not None:
        bullets.append(trend_phrase("Nasdaq", ndx.get("1M", np.nan), ndx.get("3M", np.nan), ndx.get("6M", np.nan)))
    if stoxx is not None:
        bullets.append(trend_phrase("STOXX 600", stoxx.get("1M", np.nan), stoxx.get("3M", np.nan), stoxx.get("6M", np.nan)))

    # --- Headline themes as a layer (not verbatim copying) ---
    bullets.append(f"Headline read-through: {summarize_rss_themes(rss_items)}")

    # Format
    return "\n".join([f"- {b}" for b in bullets])


def format_rss_digest(items: List[Dict[str, str]], max_items: int = 10) -> str:
    """
    Provide a *short* linked digest (not the whole feed),
    then your interpreted theme summary (above) does the heavy lifting.
    """
    if not items:
        return "_No RSS items available._"
    out = []
    for it in items[:max_items]:
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        src = it.get("source", "").strip()
        if link:
            out.append(f"- [{title}]({link}) — {src}".rstrip())
        else:
            out.append(f"- {title} — {src}".rstrip())
    return "\n".join(out)


# ----------------------------
# Movers (>=4%)
# ----------------------------
def fetch_session_movers_yahoo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    gain_urls = [
        "https://finance.yahoo.com/markets/stocks/gainers/",
        "https://finance.yahoo.com/gainers?count=100&offset=0",
    ]
    lose_urls = [
        "https://finance.yahoo.com/markets/stocks/losers/",
        "https://finance.yahoo.com/losers?count=100&offset=0",
    ]

    def pick_table(urls):
        for u in urls:
            try:
                tables = read_html_tables(u)
                if not tables:
                    continue
                return max(tables, key=lambda x: x.shape[0])
            except Exception:
                continue
        return pd.DataFrame()

    return pick_table(gain_urls), pick_table(lose_urls)


def fetch_afterhours_movers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    gain = pd.DataFrame()
    lose = pd.DataFrame()

    # Primary: Investing.com
    try:
        tables = read_html_tables("https://www.investing.com/equities/after-hours")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("%" in c for c in cols) and (any("chg" in c for c in cols) or any("change" in c for c in cols)):
                gain = t.copy()
                break
    except Exception:
        pass

    # Fallback: StockAnalysis afterhours
    if gain.empty:
        try:
            tables = read_html_tables("https://stockanalysis.com/markets/afterhours/")
            if len(tables) >= 2:
                gain = tables[0]
                lose = tables[1]
        except Exception:
            pass

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["_symbol", "_pct"])

        out = df.copy()

        pct_col = None
        for c in out.columns:
            s = str(c).lower()
            if "%" in s and ("chg" in s or "change" in s):
                pct_col = c
                break
        if pct_col is None:
            for c in out.columns:
                if "%" in str(c):
                    pct_col = c
                    break

        if pct_col is not None:
            out["_pct"] = (
                out[pct_col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            out["_pct"] = pd.to_numeric(out["_pct"], errors="coerce")
        else:
            out["_pct"] = np.nan

        sym_col = None
        for c in out.columns:
            if str(c).lower() in ("symbol", "ticker"):
                sym_col = c
                break
        if sym_col is None:
            sym_col = out.columns[0]

        out["_symbol"] = out[sym_col].astype(str).str.split().str[0]
        out = out.dropna(subset=["_pct"])
        return out[["_symbol", "_pct"]]

    # If one combined table, split by sign
    if not gain.empty and lose.empty:
        ng = normalize(gain)
        g = ng[ng["_pct"] >= 0].copy()
        l = ng[ng["_pct"] < 0].copy()
        return g, l

    return normalize(gain), normalize(lose)


def filter_movers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always returns schema: ['symbol','pct']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "pct"])

    out = df.copy()

    # after-hours normalized
    if "_pct" in out.columns:
        out["pct"] = pd.to_numeric(out["_pct"], errors="coerce")
    else:
        pct_col = None
        for c in out.columns:
            s = str(c).lower()
            if "%" in s and ("change" in s or "chg" in s):
                pct_col = c
                break
        if pct_col is None:
            for c in out.columns:
                s = str(c).lower()
                if "change" in s and "%" in s:
                    pct_col = c
                    break

        if pct_col is not None:
            out["pct"] = (
                out[pct_col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            out["pct"] = pd.to_numeric(out["pct"], errors="coerce")
        else:
            out["pct"] = np.nan

    if "_symbol" in out.columns:
        out["symbol"] = out["_symbol"].astype(str)
    else:
        sym_col = None
        for c in out.columns:
            if str(c).lower() in ("symbol", "ticker"):
                sym_col = c
                break
        if sym_col is None:
            sym_col = out.columns[0]
        out["symbol"] = out[sym_col].astype(str).str.split().str[0]

    out = out.dropna(subset=["pct"])
    out = out.loc[out["pct"].abs() >= MOVER_THRESHOLD_PCT].copy()

    if out.empty:
        return pd.DataFrame(columns=["symbol", "pct"])

    out = out.sort_values("pct", ascending=False)
    return out[["symbol", "pct"]].head(30)


def compute_universe_movers(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in ohlcv.items():
        c = df["Close"].dropna()
        if len(c) < 2:
            continue
        p = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
        if abs(p) >= MOVER_THRESHOLD_PCT:
            rows.append({"symbol": t, "pct": p})
    if not rows:
        return pd.DataFrame(columns=["symbol", "pct"])
    d = pd.DataFrame(rows).sort_values("pct", ascending=False)
    return d[["symbol", "pct"]].head(50)


# ----------------------------
# After-hours fallback for custom tickers
# ----------------------------
def after_hours_snapshot(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="2d", interval="1m", prepost=True)
        if h is None or h.empty:
            return None
        h = h.dropna()
        last_px = float(h["Close"].iloc[-1])

        idx = h.index
        mask = (idx.hour == 16) & (idx.minute <= 5)
        if mask.any():
            reg_close = float(h.loc[mask, "Close"].iloc[-1])
        else:
            prev_day = idx[-1].date() - dt.timedelta(days=1)
            prev = h[h.index.date == prev_day]
            if prev.empty:
                return None
            reg_close = float(prev["Close"].iloc[-1])

        if reg_close <= 0:
            return None
        return (last_px / reg_close - 1.0) * 100.0
    except Exception:
        return None


def compute_custom_afterhours_movers(custom: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for t in custom:
        pctv = after_hours_snapshot(t)
        if pctv is None:
            continue
        if abs(pctv) >= MOVER_THRESHOLD_PCT:
            rows.append({"symbol": t, "pct": float(pctv)})
    if not rows:
        empty = pd.DataFrame(columns=["symbol", "pct"])
        return empty, empty
    df = pd.DataFrame(rows).sort_values("pct", ascending=False)
    gain = df[df["pct"] >= 0].head(30)
    lose = df[df["pct"] < 0].sort_values("pct", ascending=True).head(30)
    return gain, lose


# ----------------------------
# Technical patterns (lightweight heuristics)
# ----------------------------
@dataclass
class LevelSignal:
    ticker: str
    signal: str
    pattern: str
    direction: str
    level: float
    close: float
    atr: float
    dist_atr: float
    pct_today: Optional[float] = None
    chart_path: Optional[str] = None


def _swing_points(series: pd.Series, window: int = 3) -> Tuple[List[int], List[int]]:
    s = series.values
    highs, lows = [], []
    for i in range(window, len(s) - window):
        seg = s[i - window:i + window + 1]
        if np.isnan(seg).any():
            continue
        if s[i] == np.max(seg):
            highs.append(i)
        if s[i] == np.min(seg):
            lows.append(i)
    return highs, lows


def _line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return (0.0, float(y[-1]) if len(y) else 0.0)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _classify_vs_level(close: float, level: float, atr_val: float, direction: str) -> Tuple[str, float]:
    if atr_val is None or atr_val <= 0 or math.isnan(atr_val):
        atr_val = max(level * 0.01, 1e-6)
    dist_atr = (close - level) / atr_val

    if direction == "BREAKOUT":
        if close >= level + ATR_CONFIRM_MULT * atr_val:
            return "CONFIRMED_", dist_atr
        if close >= level:
            return "SOFT_", dist_atr
        if (level - close) <= EARLY_MULT * atr_val:
            return "EARLY_", dist_atr
        return "", dist_atr

    # BREAKDOWN
    if close <= level - ATR_CONFIRM_MULT * atr_val:
        return "CONFIRMED_", dist_atr
    if close <= level:
        return "SOFT_", dist_atr
    if (close - level) <= EARLY_MULT * atr_val:
        return "EARLY_", dist_atr
    return "", dist_atr


def detect_hs_top(df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
    d = df.tail(LOOKBACK_DAYS).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=4)
    if len(highs_idx) < 3 or len(lows_idx) < 2:
        return None

    recent_vals = [(i, float(c.iloc[i])) for i in highs_idx[-10:]]
    last3 = recent_vals[-3:]
    if len(last3) < 3:
        return None

    (i1, p1), (i2, p2), (i3, p3) = last3
    if not (p2 > p1 and p2 > p3):
        return None
    if min(p1, p3) <= 0:
        return None
    if abs(p1 - p3) / min(p1, p3) > 0.07:
        return None

    lows_between_1 = [i for i in lows_idx if i1 < i < i2]
    lows_between_2 = [i for i in lows_idx if i2 < i < i3]
    if not lows_between_1 or not lows_between_2:
        return None
    l1 = float(c.iloc[lows_between_1[-1]])
    l2 = float(c.iloc[lows_between_2[0]])
    neckline = (l1 + l2) / 2.0
    return ("HS_TOP", "BREAKDOWN", neckline)


def detect_inverse_hs(df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
    d = df.tail(LOOKBACK_DAYS).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=4)
    if len(lows_idx) < 3 or len(highs_idx) < 2:
        return None

    recent_vals = [(i, float(c.iloc[i])) for i in lows_idx[-10:]]
    last3 = recent_vals[-3:]
    if len(last3) < 3:
        return None

    (i1, p1), (i2, p2), (i3, p3) = last3
    if not (p2 < p1 and p2 < p3):
        return None
    if min(p1, p3) <= 0:
        return None
    if abs(p1 - p3) / min(p1, p3) > 0.08:
        return None

    highs_between_1 = [i for i in highs_idx if i1 < i < i2]
    highs_between_2 = [i for i in highs_idx if i2 < i < i3]
    if not highs_between_1 or not highs_between_2:
        return None
    h1 = float(c.iloc[highs_between_1[-1]])
    h2 = float(c.iloc[highs_between_2[0]])
    neckline = (h1 + h2) / 2.0
    return ("IHS", "BREAKOUT", neckline)


def detect_structure(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
    d = df.tail(180).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=3)
    if len(highs_idx) < 4 or len(lows_idx) < 4:
        return None

    hi = np.array(highs_idx[-6:], dtype=int)
    lo = np.array(lows_idx[-6:], dtype=int)

    xh = hi.astype(float)
    yh = np.array([float(c.iloc[i]) for i in hi])
    xl = lo.astype(float)
    yl = np.array([float(c.iloc[i]) for i in lo])

    a_u, b_u = _line_fit(xh, yh)
    a_l, b_l = _line_fit(xl, yl)

    x_last = float(len(c) - 1)
    upper_now = a_u * x_last + b_u
    lower_now = a_l * x_last + b_l
    if not (upper_now > lower_now):
        return None

    x_early = float(max(0, len(c) - 60))
    width_now = (a_u * x_last + b_u) - (a_l * x_last + b_l)
    width_then = (a_u * x_early + b_u) - (a_l * x_early + b_l)
    if width_then <= 0:
        return None

    converging = width_now < 0.75 * width_then
    diverging = width_now > 1.15 * width_then

    if abs(a_u) < 0.02 and abs(a_l) < 0.02 and (width_now / max(lower_now, 1e-6)) < 0.18:
        return ("RECT", float(upper_now), float(lower_now))

    if diverging:
        return ("BROADEN", float(upper_now), float(lower_now))

    if converging:
        if a_u < 0 and a_l > 0:
            return ("TRIANGLE", float(upper_now), float(lower_now))
        if a_u < 0 and a_l < 0:
            return ("WEDGE_DOWN", float(upper_now), float(lower_now))
        if a_u > 0 and a_l > 0:
            return ("WEDGE_UP", float(upper_now), float(lower_now))
        return ("TRIANGLE", float(upper_now), float(lower_now))

    return None


def compute_signals_for_ticker(ticker: str, df: pd.DataFrame) -> List[LevelSignal]:
    sigs: List[LevelSignal] = []
    if df is None or df.empty or len(df) < 80:
        return sigs

    d = df.dropna(subset=["Close"]).copy()
    if len(d) < 80:
        return sigs

    a = atr(d, ATR_N)
    atr_val = float(a.dropna().iloc[-1]) if not a.dropna().empty else float("nan")
    close = float(d["Close"].iloc[-1])
    pct_today = pct_change_last(d)

    hs = detect_hs_top(d)
    if hs:
        pattern, direction, level = hs
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction)
        if prefix:
            sigs.append(LevelSignal(
                ticker=ticker, signal=f"{prefix}{pattern}_{direction}",
                pattern=pattern, direction=direction,
                level=float(level), close=close, atr=atr_val,
                dist_atr=float(dist_atr), pct_today=pct_today
            ))

    ihs = detect_inverse_hs(d)
    if ihs:
        pattern, direction, level = ihs
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction)
        if prefix:
            sigs.append(LevelSignal(
                ticker=ticker, signal=f"{prefix}{pattern}_{direction}",
                pattern=pattern, direction=direction,
                level=float(level), close=close, atr=atr_val,
                dist_atr=float(dist_atr), pct_today=pct_today
            ))

    st = detect_structure(d)
    if st:
        pattern, upper_level, lower_level = st

        prefix_u, dist_u = _classify_vs_level(close, upper_level, atr_val, "BREAKOUT")
        if prefix_u:
            sigs.append(LevelSignal(
                ticker=ticker, signal=f"{prefix_u}{pattern}_BREAKOUT",
                pattern=pattern, direction="BREAKOUT",
                level=float(upper_level), close=close, atr=atr_val,
                dist_atr=float(dist_u), pct_today=pct_today
            ))

        prefix_l, dist_l = _classify_vs_level(close, lower_level, atr_val, "BREAKDOWN")
        if prefix_l:
            sigs.append(LevelSignal(
                ticker=ticker, signal=f"{prefix_l}{pattern}_BREAKDOWN",
                pattern=pattern, direction="BREAKDOWN",
                level=float(lower_level), close=close, atr=atr_val,
                dist_atr=float(dist_l), pct_today=pct_today
            ))

    return sigs


# ----------------------------
# Charting
# ----------------------------
def plot_signal_chart(ticker: str, df: pd.DataFrame, sig: LevelSignal) -> Optional[str]:
    if df is None or df.empty:
        return None
    d = df.tail(220).dropna(subset=["Close"]).copy()
    if len(d) < 60:
        return None

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(d.index, d["Close"].values)
    ax.axhline(sig.level, linestyle="--")

    title = f"{ticker} | {sig.signal} | level={sig.level:.2f} | dist={sig.dist_atr:+.2f} ATR"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")

    ax.scatter([d.index[-1]], [d["Close"].iloc[-1]])
    ax.text(d.index[-1], d["Close"].iloc[-1], f"  {d['Close'].iloc[-1]:.2f}", va="center")

    fname = f"{ticker}_{sig.signal}.png"
    fname = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", fname)
    out_path = IMG_DIR / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return f"img/{fname}"


# ----------------------------
# Reporting utilities
# ----------------------------
def signals_to_df(signals: List[LevelSignal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(columns=["Ticker", "Signal", "Pattern", "Dir", "Close", "Level", "Dist(ATR)", "Day%", "Chart"])
    return pd.DataFrame([{
        "Ticker": s.ticker,
        "Signal": s.signal,
        "Pattern": s.pattern,
        "Dir": s.direction,
        "Close": s.close,
        "Level": s.level,
        "Dist(ATR)": s.dist_atr,
        "Day%": s.pct_today if s.pct_today is not None else np.nan,
        "Chart": s.chart_path or ""
    } for s in signals])


def md_table_from_df(df: pd.DataFrame, cols: List[str], max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_None_"
    d = df.copy().head(max_rows)
    for c in ["Close", "Level"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "Dist(ATR)" in d.columns:
        d["Dist(ATR)"] = pd.to_numeric(d["Dist(ATR)"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    if "Day%" in d.columns:
        d["Day%"] = pd.to_numeric(d["Day%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "Chart" in d.columns:
        d["Chart"] = d["Chart"].apply(lambda p: f"[chart]({p})" if isinstance(p, str) and p else "")
    return d[cols].to_markdown(index=False)


def diff_new_ended(prev: Dict[str, List[str]], cur: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    prev_set = set(prev.get("signals", []))
    cur_set = set(cur.get("signals", []))
    return sorted(cur_set - prev_set), sorted(prev_set - cur_set)


def movers_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title}:** _None ≥ {MOVER_THRESHOLD_PCT:.0f}%_\n"
    t = df.copy()
    t["pct"] = pd.to_numeric(t["pct"], errors="coerce").map(lambda x: f"{x:+.2f}%")
    return f"**{title}:**\n\n" + t[["symbol", "pct"]].to_markdown(index=False) + "\n"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "custom"], default=os.environ.get("MODE", "full"))
    ap.add_argument("--max-tickers", type=int, default=int(os.environ.get("MAX_TICKERS", "0")))
    args = ap.parse_args()

    custom = get_custom_tickers()
    spx = get_sp500_tickers()
    ndx = get_nasdaq100_tickers()

    # Universe for technical scan & universe-mover fallback
    if args.mode == "custom":
        universe = custom
    else:
        universe = sorted(set(custom + spx + ndx))

    if args.max_tickers and args.max_tickers > 0:
        universe = universe[:args.max_tickers]

    now = dt.datetime.now(dt.timezone.utc)
    header_time = now.astimezone().strftime("%Y-%m-%d %H:%M %Z")

    # RSS (for themes + catalyst digest)
    rss_items = fetch_rss_headlines(limit_total=14)

    # 1) Cross-asset snapshot (multi-horizon)
    snapshot_df = fetch_market_snapshot_multi()

    # Download OHLCV once (for technicals + universe mover fallback)
    ohlcv = yf_download_chunk(universe)

    # 2) Movers
    session_g, session_l = fetch_session_movers_yahoo()
    session_gf = filter_movers(session_g)
    session_lf = filter_movers(session_l)
    if not session_lf.empty:
        session_lf = session_lf.sort_values("pct", ascending=True)

    # Fallback: compute from our universe
    if session_gf.empty and session_lf.empty:
        uni_movers = compute_universe_movers(ohlcv)
        session_gf = uni_movers[uni_movers["pct"] >= 0].head(30)
        session_lf = uni_movers[uni_movers["pct"] < 0].sort_values("pct", ascending=True).head(30)

    ah_g, ah_l = fetch_afterhours_movers()
    ah_gf = filter_movers(ah_g)
    ah_lf = filter_movers(ah_l)
    if not ah_lf.empty:
        ah_lf = ah_lf.sort_values("pct", ascending=True)

    # Fallback after-hours: custom tickers via pre/post snapshot
    if ah_gf.empty and ah_lf.empty and custom:
        fg, fl = compute_custom_afterhours_movers(custom)
        ah_gf, ah_lf = fg, fl

    # 4) Technical triggers
    all_signals: List[LevelSignal] = []
    for t in universe:
        df = ohlcv.get(t)
        if df is None or df.empty:
            continue
        all_signals.extend(compute_signals_for_ticker(t, df))

    early = [s for s in all_signals if s.signal.startswith("EARLY_")]
    triggered = [s for s in all_signals if s.signal.startswith("SOFT_") or s.signal.startswith("CONFIRMED_")]

    def rank_trigger(s: LevelSignal) -> Tuple[int, float]:
        tier = 0 if s.signal.startswith("CONFIRMED_") else (1 if s.signal.startswith("SOFT_") else 2)
        return (tier, abs(s.dist_atr))

    triggered_sorted = sorted(triggered, key=rank_trigger)
    early_sorted = sorted(early, key=lambda s: abs(s.dist_atr))

    # Charts
    for s in triggered_sorted[:MAX_CHARTS_TRIGGERED]:
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
    for s in early_sorted[:MAX_CHARTS_EARLY]:
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)

    # State diff
    state = load_state()
    prev = {"signals": state.get("signals", [])}
    cur_ids = [f"{s.ticker}|{s.signal}" for s in all_signals]
    state["signals"] = cur_ids
    save_state(state)
    new_ids, ended_ids = diff_new_ended(prev, {"signals": cur_ids})
    new_set = set(new_ids)

    def mark_new(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            return df, df
        d = df.copy()
        d["_id"] = d["Ticker"].astype(str) + "|" + d["Signal"].astype(str)
        d_new = d[d["_id"].isin(new_set)].drop(columns=["_id"])
        d_old = d[~d["_id"].isin(new_set)].drop(columns=["_id"])
        return d_new, d_old

    df_early = signals_to_df(early_sorted)
    df_trig = signals_to_df(triggered_sorted)
    df_early_new, df_early_old = mark_new(df_early)
    df_trig_new, df_trig_old = mark_new(df_trig)

    # Assemble markdown
    md: List[str] = []
    md.append("# Daily Report\n")
    md.append(f"_Generated: **{header_time}**_\n")

    # 1) Market recap + positioning
    md.append("## 1) Market recap & positioning\n")
    md.append("**Key tape (multi-horizon):**\n")
    md.append(format_snapshot_table_multi(snapshot_df))
    md.append("")
    md.append("**Commentary (interpretation):**\n")
    md.append(build_commentary(snapshot_df, rss_items))
    md.append("")

    # 2) Movers
    md.append("## 2) Biggest movers (≥ 4%)\n")
    md.append(movers_table(session_gf, "Session gainers"))
    md.append(movers_table(session_lf, "Session losers"))
    md.append(movers_table(ah_gf, "After-hours gainers"))
    md.append(movers_table(ah_lf, "After-hours losers"))

    # 4) Technical triggers
    md.append("## 4) Technical triggers\n")
    md.append(f"**Breakout confirmation rule:** close beyond trigger by **≥ {ATR_CONFIRM_MULT:.1f} ATR**.\n")

    md.append("### 4A) Early callouts (~80% complete)\n")
    md.append("_Close enough to pre-plan. “Close enough” = within 0.5 ATR of neckline/boundary._\n")
    md.append("**NEW (today):**\n")
    md.append(md_table_from_df(df_early_new, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=40))
    md.append("\n**ONGOING:**\n")
    md.append(md_table_from_df(df_early_old, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=80))
    md.append("")

    md.append("### 4B) Breakouts / breakdowns (or about to)\n")
    md.append("_Includes **SOFT** (pierced but <0.5 ATR) and **CONFIRMED** (≥0.5 ATR)._ \n")
    md.append("**NEW (today):**\n")
    md.append(md_table_from_df(df_trig_new, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=60))
    md.append("\n**ONGOING:**\n")
    md.append(md_table_from_df(df_trig_old, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=120))
    md.append("")

    # 5) Catalysts (RSS digest + theme summary already included in commentary)
    md.append("## 5) Needle-moving catalysts (RSS digest)\n")
    md.append("_Digest is linked for quick drill-down; the interpretation is in Section 1 “Headline read-through”._\n")
    md.append(format_rss_digest(rss_items, max_items=10))
    md.append("")

    # Changelog
    md.append("## Changelog\n")
    if new_ids:
        md.append("**New signals:**\n")
        for x in new_ids[:120]:
            md.append(f"- {x}")
    else:
        md.append("**New signals:** _None_\n")

    if ended_ids:
        md.append("\n**Ended signals:**\n")
        for x in ended_ids[:120]:
            md.append(f"- {x}")
    else:
        md.append("\n**Ended signals:** _None_\n")

    md_text = "\n".join(md).strip() + "\n"

    # Write outputs
    write_text(REPORT_PATH, md_text)
    write_text(INDEX_PATH, md_text)

    print(f"Wrote: {REPORT_PATH}")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Universe={len(universe)}  Signals: early={len(early_sorted)} triggered={len(triggered_sorted)}")


if __name__ == "__main__":
    try:
        main()
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        err = traceback.format_exc()
        print(err)
        fallback = (
            "# Daily Report\n\n"
            "## ERROR\n\n"
            "The run crashed. Traceback:\n\n"
            "```text\n"
            f"{err[-4000:]}\n"
            "```\n"
        )
        write_text(REPORT_PATH, fallback)
        write_text(INDEX_PATH, fallback)
        raise SystemExit(0)
