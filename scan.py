#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily Ticker Report (GitHub Pages)

Latest changes requested:
- Market recap: Executive summary FIRST (no “max 2 sentences” label)
- Replace “risk-on” phrasing with plain-English interpretation (e.g., “Markets rebounded as AI fears eased…”)
- Snapshot “Last” formatting standardized: thousands separator comma + 2 decimals (e.g., 25,020.93)
- Remove 🟩🟥 squares; keep only colored % text

Also already applied:
- Drop WTI, DXY, US 10Y from cross-asset tape
- VIX + EUR/USD: 5Y Google-Finance-like card images

NEW (this update):
- Force RIGHT alignment for numeric figure columns in all markdown tables
  (Key tape, Movers, Technical trigger tables) by patching the markdown
  alignment row to use ---: on numeric columns.
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
# Default watchlist (user-defined)
# ----------------------------
# Ensures your full 44-ticker watchlist is ALWAYS included when MODE=custom.
# You can disable this by setting USE_DEFAULT_WATCHLIST=0 in the environment.
WATCHLIST_44: List[str] = [
    "MELI","ARM","QBTS","IONQ","HOOD","PLTR","SNPS","AVGO","CDNS","AMAT",
    "NFLX","LRCX","TSM","DASH","ISRG","MUV2.DE","PGR","CMG","ANF","DECK",
    "NU","UCG.MI","MC.PA","RMS.PA","VST","OKLO","SMR","CEG","LEU","CCJ",
    "000660.KS","NVDA","NVO","LLY","AMZN","GOOGL","AAPL","META","MSFT","ASML",
    "WMT","BYDDY","RRTL","ARR",
]


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
EMAIL_MD_PATH = DOCS_DIR / "email.md"
EMAIL_TXT_PATH = DOCS_DIR / "email.txt"

CUSTOM_TICKERS_PATH = CONFIG_DIR / "tickers_custom.txt"
SP500_LOCAL = CONFIG_DIR / "universe_sp500.txt"
NDX_LOCAL = CONFIG_DIR / "universe_nasdaq100.txt"


# ----------------------------
# Config knobs
# ----------------------------
MOVER_THRESHOLD_PCT = 4.0

ATR_N = 14
ATR_CONFIRM_MULT = 0.5     # confirmed breakout/breakdown threshold
EARLY_MULT = 0.5           # early callout threshold (within 0.5 ATR)

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
# Helpers: Markdown table alignment
# ----------------------------
def _patch_markdown_alignment(md: str, aligns: Tuple[str, ...]) -> str:
    """
    GitHub-flavored markdown aligns columns based on the header separator row:
      left:  :---
      right: ---:
      center::---:

    Pandas' to_markdown may not always emit alignment; this function forces it.
    """
    if not md or not isinstance(md, str):
        return md
    lines = md.splitlines()
    if len(lines) < 2:
        return md

    # Pandas markdown tables: line0 header, line1 separator, then rows.
    # Only patch if the table has the expected pipe structure.
    if "|" not in lines[0] or "|" not in lines[1]:
        return md

    # Ensure column count matches aligns length
    # Count columns by splitting header line on | and removing empties.
    header_cols = [c.strip() for c in lines[0].split("|") if c.strip() != ""]
    if len(header_cols) != len(aligns):
        return md

    sep = []
    for a in aligns:
        a = (a or "").lower()
        if a == "left":
            sep.append(":---")
        elif a == "right":
            sep.append("---:")
        elif a == "center":
            sep.append(":---:")
        else:
            sep.append("---")

    lines[1] = "| " + " | ".join(sep) + " |"
    return "\n".join(lines)


def df_to_markdown_aligned(df: pd.DataFrame, aligns: Tuple[str, ...], index: bool = False) -> str:
    """
    Generate markdown and force alignment row regardless of pandas/tabulate version.
    """
    md = df.to_markdown(index=index)
    return _patch_markdown_alignment(md, aligns)


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
    Minimal RSS/Atom parser returning dict(title, link, pubDate, source).

    Supports:
      - RSS <item>
      - Atom <entry>
    """
    try:
        xml_text = fetch_url_text(url, timeout=30)
        root = ET.fromstring(xml_text)

        def norm(s: Optional[str]) -> str:
            return (s or "").strip()

        items: List[Dict[str, str]] = []

        # RSS
        for item in root.findall(".//item"):
            title = norm(item.findtext("title"))
            link = norm(item.findtext("link"))
            pub = norm(item.findtext("pubDate"))
            if title:
                items.append({"title": title, "link": link, "pubDate": pub, "source": source_name})

        # Atom
        if not items:
            for entry in root.findall(".//{*}entry"):
                title = norm(entry.findtext("{*}title"))
                pub = norm(entry.findtext("{*}updated")) or norm(entry.findtext("{*}published"))
                link = ""
                # Atom links are usually attributes
                for l in entry.findall("{*}link"):
                    href = (l.attrib.get("href") or "").strip()
                    rel = (l.attrib.get("rel") or "alternate").strip()
                    if href and (rel == "alternate" or not link):
                        link = href
                # Some Atom feeds use <link>text</link>
                if not link:
                    link = norm(entry.findtext("{*}link"))
                if title:
                    items.append({"title": title, "link": link, "pubDate": pub, "source": source_name})

        return items[:limit] if items else []
    except Exception:
        return []



def fetch_rss_headlines(limit_total: int = 16) -> List[Dict[str, str]]:
    # A diversified set of free-to-read headline feeds (some articles may still have partial paywalls,
    # but headlines + links remain useful for daily context).
    feeds = [
        ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
        ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("CNBC Markets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
        ("Reuters Top News", "https://feeds.reuters.com/reuters/topNews"),
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch Top Stories", "https://feeds.marketwatch.com/marketwatch/topstories"),
        ("Financial Times", "https://www.ft.com/?format=rss"),
    ]
    all_items: List[Dict[str, str]] = []
    for name, url in feeds:
        all_items.extend(parse_rss(url, name, limit=10))

    # De-dupe by title (case-insensitive)
    seen = set()
    uniq: List[Dict[str, str]] = []
    for it in all_items:
        t = (it.get("title", "") or "").strip()
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
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
            if "ticker" in cols or "symbol" in cols:
                df = t
                break
        if df is None:
            df = tables[0]

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
    tickers = {_clean_ticker(x) for x in read_lines(CUSTOM_TICKERS_PATH)}
    # Always include the default 44-ticker watchlist unless explicitly disabled.
    if os.environ.get("USE_DEFAULT_WATCHLIST", "1").strip().lower() not in ("0", "false", "no"):
        tickers.update(WATCHLIST_44)

    extra = os.environ.get("EXTRA_TICKERS", "").strip()
    if extra:
        for x in re.split(r"[,\s]+", extra):
            x = x.strip()
            if x:
                tickers.add(_clean_ticker(x))

    return sorted(tickers)


# ----------------------------
# Market data (yfinance) - robust extraction
# ----------------------------
def extract_ohlcv_from_download(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None

    # Single ticker: flat columns
    if not isinstance(data.columns, pd.MultiIndex):
        if not {"Open", "High", "Low", "Close"}.issubset(set(data.columns)):
            return None
        df = data.copy()
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
# Snapshot: cross-asset + multi-horizon (NO WTI / DXY / 10Y)
# ----------------------------
def _extract_close_series(download_df: pd.DataFrame, sym: str) -> Optional[pd.Series]:
    if download_df is None or download_df.empty:
        return None
    if not isinstance(download_df.columns, pd.MultiIndex):
        return download_df["Close"].dropna() if "Close" in download_df.columns else None

    cols = download_df.columns
    if ("Close", sym) in cols:
        return download_df[("Close", sym)].dropna()
    if (sym, "Close") in cols:
        return download_df[(sym, "Close")].dropna()
    return None


def _color_pct_cell(x: float) -> str:
    """
    No emojis/squares. Colored % only.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if x > 0:
        return f'<span style="color:#11823b;">{x:+.2f}%</span>'
    if x < 0:
        return f'<span style="color:#b91c1c;">{x:+.2f}%</span>'
    return f'<span style="color:#6b7280;">{x:+.2f}%</span>'


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
    Instruments requested:
    - US: Nasdaq 100, S&P 500 (plus QQQ/SPY)
    - Europe: STOXX Europe 600, DAX, CAC 40, FTSE 100
    - Risk: VIX
    - FX: EUR/USD
    - Commodities: WTI Crude, Gold, Silver, Coffee, Cocoa
    - Crypto: Bitcoin
    """
    instruments = [
        ("Nasdaq 100", "^NDX"),
        ("S&P 500", "^GSPC"),
        ("QQQ", "QQQ"),
        ("SPY", "SPY"),

        ("STOXX Europe 600", "^STOXX"),
        ("DAX", "^GDAXI"),
        ("CAC 40", "^FCHI"),
        ("FTSE 100", "^FTSE"),

        ("VIX", "^VIX"),
        ("EUR/USD", "EURUSD=X"),

        ("WTI Crude", "CL=F"),

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

    # Standardize Last: thousands comma + 2 decimals
    d["Last"] = pd.to_numeric(d["Last"], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    for c in ["1D", "7D", "1M", "3M", "6M"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").map(_color_pct_cell)

    cols = ["Instrument", "Last", "1D", "7D", "1M", "3M", "6M"]
    out = d[cols]

    # Force alignment: first column left, rest right
    aligns = ("left",) + tuple("right" for _ in cols[1:])
    return df_to_markdown_aligned(out, aligns=aligns, index=False)


# ----------------------------
# Google-Finance-like card charts (5Y): VIX and EUR/USD
# ----------------------------
def _fmt_de(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{decimals}f}"
    # 1,234.56 -> 1.234,56
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def _fmt_de_signed(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    sign = "+" if x > 0 else ""
    return f"{sign}{_fmt_de(x, decimals)}"


def _fmt_de_date(ts: pd.Timestamp) -> str:
    months = {
        1: "Jan.", 2: "Feb.", 3: "Mär.", 4: "Apr.", 5: "Mai", 6: "Jun.",
        7: "Jul.", 8: "Aug.", 9: "Sep.", 10: "Okt.", 11: "Nov.", 12: "Dez."
    }
    ts = pd.Timestamp(ts)
    return f"{ts.day}. {months.get(ts.month, ts.strftime('%b'))} {ts.year}"


def plot_gf_card_5y(
    symbol: str,
    title: str,
    subtitle: str,
    out_name: str,
    decimals_last: int = 2,
    line_color: str = "#d93025",
) -> Optional[str]:
    """
    Static image that mimics the Google Finance card:
    - Title + subtitle
    - Big last value + daily change (red/green)
    - 5Y line chart + max spike marker + label box
    - Footer: Open/High/Low/Prev + 52W high/low
    """
    try:
        data = yf.download(
            tickers=[symbol],
            period="5y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        df = extract_ohlcv_from_download(data, symbol)
        if df is None or df.empty or df["Close"].dropna().empty:
            return None

        df = df.dropna(subset=["Close"]).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
        chg = last - prev
        chg_pct = (chg / prev * 100.0) if prev != 0 else float("nan")

        o = float(df["Open"].iloc[-1]) if "Open" in df.columns and pd.notna(df["Open"].iloc[-1]) else float("nan")
        h = float(df["High"].iloc[-1]) if "High" in df.columns and pd.notna(df["High"].iloc[-1]) else float("nan")
        l = float(df["Low"].iloc[-1]) if "Low" in df.columns and pd.notna(df["Low"].iloc[-1]) else float("nan")

        df_52w = df.tail(252)
        hi_52 = float(df_52w["High"].max()) if "High" in df_52w.columns else float(df_52w["Close"].max())
        lo_52 = float(df_52w["Low"].min()) if "Low" in df_52w.columns else float(df_52w["Close"].min())

        s = df["Close"].dropna()
        max_idx = int(np.nanargmax(s.values))
        max_dt = s.index[max_idx]
        max_val = float(s.iloc[max_idx])

        change_color = "#188038" if chg >= 0 else "#d93025"

        fig = plt.figure(figsize=(12.5, 7.0))
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 4.2, 1.1], hspace=0.18)

        ax_head = fig.add_subplot(gs[0, 0]); ax_head.axis("off")
        ax = fig.add_subplot(gs[1, 0])
        ax_foot = fig.add_subplot(gs[2, 0]); ax_foot.axis("off")

        # Header
        ax_head.text(0.00, 0.78, title, fontsize=24, fontweight="bold", ha="left", va="center")
        ax_head.text(0.00, 0.38, subtitle, fontsize=12.5, color="#5f6368", ha="left", va="center")

        ax_head.text(0.00, -0.05, _fmt_de(last, decimals_last), fontsize=44, fontweight="bold",
                     ha="left", va="center")
        ax_head.text(0.00, -0.55,
                     f"{_fmt_de_signed(chg, decimals_last)} ({_fmt_de_signed(chg_pct, 2)}%)",
                     fontsize=16, color=change_color, ha="left", va="center")
        ax_head.text(0.00, -0.92, f"{_fmt_de_date(df.index[-1])}",
                     fontsize=11.5, color="#5f6368", ha="left", va="center")

        # Chart
        ax.plot(s.index, s.values, color=line_color, linewidth=2.2)
        ax.grid(True, axis="y", alpha=0.18)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#dadce0")
        ax.spines["bottom"].set_color("#dadce0")
        ax.tick_params(axis="x", colors="#5f6368")
        ax.tick_params(axis="y", colors="#5f6368")

        ax.scatter([max_dt], [max_val], s=60, color=line_color, zorder=4)
        label = f"{_fmt_de(max_val, decimals_last)}  {_fmt_de_date(max_dt)}"
        ax.annotate(
            label,
            xy=(max_dt, max_val),
            xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dadce0"),
            fontsize=10.5,
            color="#202124",
        )

        y_min = float(np.nanmin(s.values))
        y_max = float(np.nanmax(s.values))
        pad = (y_max - y_min) * 0.10 if y_max > y_min else max(1.0, y_max * 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)

        # Footer stats
        stats_left = [
            ("Eröffnung", _fmt_de(o, decimals_last)),
            ("Hoch", _fmt_de(h, decimals_last)),
            ("Tief", _fmt_de(l, decimals_last)),
            ("Vort.Schl.", _fmt_de(prev, decimals_last)),
        ]
        stats_right = [
            ("52-Wo-Hoch", _fmt_de(hi_52, decimals_last)),
            ("52-Wo-Tief", _fmt_de(lo_52, decimals_last)),
        ]

        x0, y0 = 0.00, 0.75
        dx, dy = 0.24, 0.38
        for i, (k, v) in enumerate(stats_left):
            ax_foot.text(x0 + (i % 2) * dx, y0 - (i // 2) * dy, k, fontsize=11.5, color="#5f6368", ha="left")
            ax_foot.text(x0 + (i % 2) * dx + 0.12, y0 - (i // 2) * dy, v, fontsize=12.5, color="#202124", ha="left")

        x1 = 0.62
        for i, (k, v) in enumerate(stats_right):
            ax_foot.text(x1, y0 - i * dy, k, fontsize=11.5, color="#5f6368", ha="left")
            ax_foot.text(x1 + 0.18, y0 - i * dy, v, fontsize=12.5, color="#202124", ha="left")

        fig.tight_layout()
        out_path = IMG_DIR / out_name
        fig.savefig(out_path, dpi=175)
        plt.close(fig)

        return f"img/{out_name}"

    except Exception:
        return None


# ----------------------------
# Executive summary (plain-English, no “risk-on”)
# ----------------------------
def summarize_rss_themes(items: List[Dict[str, str]]) -> str:
    if not items:
        return "no reliable RSS feed at runtime"

    text = " ".join([it.get("title", "") for it in items]).lower()
    themes = []

    if any(k in text for k in ["fed", "fomc", "minutes", "powell", "rates", "yield", "treasury", "inflation"]):
        themes.append("Fed/rates")
    if any(k in text for k in ["ai", "chip", "semiconductor", "nvidia", "software", "cloud"]):
        themes.append("AI/tech")
    if any(k in text for k in ["earnings", "guidance", "forecast", "results"]):
        themes.append("earnings/guidance")
    if any(k in text for k in ["oil", "energy", "geopolitic", "war"]):
        themes.append("macro/geopolitics")

    if not themes:
        themes.append("mixed macro")

    return ", ".join(themes[:3])




def _openai_responses_exec_summary(payload_text: str) -> Optional[str]:
    """Call OpenAI Responses API to generate a 2–3 sentence executive summary.

    Uses GPT-5.2 pro by default (set OPENAI_MODEL to override).
    Set OPENAI_API_KEY in the environment (GitHub Actions secret).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-5.2-pro").strip() or "gpt-5.2-pro"
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "medium").strip() or "medium"

    instructions = (
        "Write the Executive summary for a daily market report. "
        "Output EXACTLY 2 or 3 sentences (no bullets, no headings).\n"
        "Requirements:\n"
        "1) Sentence 1: summarize what drove markets today using ONLY the provided headlines as evidence. "
        "If no clear catalyst is present, say so; do not invent events.\n"
        "2) Include the concrete figures that back it (at least NDX 1D, S&P 1D, VIX 1D; add one more relevant metric if helpful).\n"
        "3) Add 3–4 week context using the provided 1M vs 3M returns.\n"
        "4) Mention any watchlist mover(s) >4% incl. after-hours; if none: 'No watchlist names moved >4% incl. after-hours.'\n"
        "Style: plain English, compact, no jargon like 'risk-on', no hype."
    )

    body = {
        "model": model,
        "instructions": instructions,
        "input": payload_text,
        "temperature": 0.3,
        "max_output_tokens": 220,
        "reasoning": {"effort": effort},
        "text": {"verbosity": "low"},
    }

    try:
        req = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        with urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)

        out_text = ""
        if isinstance(data, dict):
            if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                out_text = data["output_text"].strip()
            else:
                outs = data.get("output", [])
                if isinstance(outs, list):
                    parts = []
                    for item in outs:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "message":
                            content = item.get("content", [])
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                        t = c.get("text") or ""
                                        if isinstance(t, str) and t.strip():
                                            parts.append(t.strip())
                    out_text = " ".join(parts).strip()

        out_text = re.sub(r"\s+", " ", out_text).strip()
        if not out_text:
            return None

        # Hard cap to 3 sentences
        sentences = re.split(r"(?<=[\.\!\?])\s+", out_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 3:
            out_text = " ".join(sentences[:3]).strip()
        return out_text

    except Exception:
        return None


def _absolutize_md_links(md: str, base_url: str) -> str:
    """Rewrite relative links (img/...) to absolute URLs for email rendering."""
    base_url = (base_url or "").strip()
    if not base_url:
        return md
    base = base_url.rstrip("/")
    md = re.sub(r"\]\(img/", f"]({base}/img/", md)
    return md


def write_email_assets(
    header_time: str,
    exec_summary: str,
    report_md: str,
    base_url: str,
    watchlist_movers: Dict[str, List[Tuple[str, float]]],
    new_ids: List[str],
    ended_ids: List[str],
) -> None:
    """Create docs/email.md and docs/email.txt for the workflow email step."""
    email_md = _absolutize_md_links(report_md, base_url)
    write_text(EMAIL_MD_PATH, email_md)

    def fmt_movers(items: List[Tuple[str, float]]) -> str:
        if not items:
            return "None"
        return ", ".join([f"{t} ({p:+.2f}%)" for t, p in items])

    link = f"{base_url.rstrip('/')}/report.md" if base_url else ""
    lines = []
    lines.append(f"Daily Ticker Report — {header_time}")
    lines.append("")
    lines.append("Executive summary:")
    lines.append(exec_summary.strip())
    lines.append("")
    if link:
        lines.append(f"Full report: {link}")
        lines.append("")
    lines.append("Watchlist movers (>|4%|, incl. after-hours):")
    lines.append(f"Session: {fmt_movers(watchlist_movers.get('session', []))}")
    lines.append(f"After-hours: {fmt_movers(watchlist_movers.get('after_hours', []))}")
    lines.append("")
    lines.append("New signals (today):")
    if new_ids:
        for s in new_ids[:25]:
            lines.append(f"- {s}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("Ended signals (today):")
    if ended_ids:
        for s in ended_ids[:25]:
            lines.append(f"- {s}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("Note: Full report.md is attached.")
    write_text(EMAIL_TXT_PATH, "\n".join(lines).strip() + "\n")


def build_exec_summary(
    snapshot_df: pd.DataFrame,
    rss_items: List[Dict[str, str]],
    watchlist_movers: Dict[str, List[Tuple[str, float]]],
) -> str:
    """Executive summary (2–3 sentences).

    Prefer GPT-5.2 pro prose via OpenAI API; fall back to deterministic text if API missing/fails.
    """
    if snapshot_df is None or snapshot_df.empty:
        return "Market summary unavailable (snapshot empty)."

    def row(name: str) -> Optional[pd.Series]:
        x = snapshot_df.loc[snapshot_df["Instrument"] == name]
        return None if x.empty else x.iloc[0]

    ndx = row("Nasdaq 100")
    spx = row("S&P 500")
    vix = row("VIX")
    stx = row("STOXX Europe 600")
    dax = row("DAX")
    eur = row("EUR/USD")
    gold = row("Gold")
    btc = row("Bitcoin")

    def f(r: Optional[pd.Series], key: str) -> float:
        try:
            if r is None:
                return float("nan")
            return float(r.get(key, np.nan))
        except Exception:
            return float("nan")

    # Build compact payload for the model (keeps cost low)
    top_headlines = [it.get("title","").strip() for it in (rss_items or [])][:10]
    payload = {
        "today": {
            "NDX": {"1D": f(ndx,"1D"), "1M": f(ndx,"1M"), "3M": f(ndx,"3M")},
            "S&P": {"1D": f(spx,"1D"), "1M": f(spx,"1M"), "3M": f(spx,"3M")},
            "VIX": {"1D": f(vix,"1D"), "1M": f(vix,"1M"), "3M": f(vix,"3M")},
            "STOXX": {"1D": f(stx,"1D")} if stx is not None else None,
            "DAX": {"1D": f(dax,"1D")} if dax is not None else None,
            "EURUSD": {"1D": f(eur,"1D")} if eur is not None else None,
            "Gold": {"1D": f(gold,"1D")} if gold is not None else None,
            "BTC": {"1D": f(btc,"1D")} if btc is not None else None,
        },
        "watchlist_movers_over_4pct": {
            "session": watchlist_movers.get("session", []),
            "after_hours": watchlist_movers.get("after_hours", []),
        },
        "headlines": top_headlines,
    }

    payload_text = json.dumps(payload, ensure_ascii=False)

    llm = _openai_responses_exec_summary(payload_text)
    if llm:
        return llm

    # Deterministic fallback (still 2 sentences + movers mention)
    themes = summarize_rss_themes(rss_items)
    s1 = (
        f"Markets moved on the day with the Nasdaq leading (NDX {f(ndx,'1D'):+.2f}% vs S&P {f(spx,'1D'):+.2f}%) and volatility at VIX {f(vix,'1D'):+.2f}%, with headlines clustering around {themes}."
    )
    s2 = (
        f"Over the past month vs ~3 months, S&P {f(spx,'1M'):+.2f}% vs {f(spx,'3M'):+.2f}% and NDX {f(ndx,'1M'):+.2f}% vs {f(ndx,'3M'):+.2f}% frame today’s move in context; "
        f"{'No watchlist names moved >4% incl. after-hours.' if (not watchlist_movers.get('session') and not watchlist_movers.get('after_hours')) else 'Watchlist movers >4%: ' + ', '.join([t for t,_ in (watchlist_movers.get('session', []) + watchlist_movers.get('after_hours', []))][:6]) + '.'}"
    )
    return s1 + " " + s2


def format_rss_digest(items: List[Dict[str, str]], max_items: int = 10) -> str:
    if not items:
        return "_No RSS items available._"
    out = []
    for it in items[:max_items]:
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        src = it.get("source", "").strip()
        if link:
            out.append(f"- [{title}]({link}) — {src}")
        else:
            out.append(f"- {title} — {src}")
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

    return normalize(gain), normalize(lose)


def filter_movers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always returns schema ['symbol','pct'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "pct"])

    out = df.copy()

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
            out["pct"] = np.nan
        else:
            out["pct"] = (
                out[pct_col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            out["pct"] = pd.to_numeric(out["pct"], errors="coerce")

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


def movers_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title}:** _None ≥ {MOVER_THRESHOLD_PCT:.0f}%_\n"
    t = df.copy()
    t["pct"] = pd.to_numeric(t["pct"], errors="coerce").map(lambda x: f"{x:+.2f}%")
    out = t[["symbol", "pct"]]
    md = df_to_markdown_aligned(out, aligns=("left", "right"), index=False)
    return f"**{title}:**\n\n" + md + "\n"

# ----------------------------
# Earnings calendar (watchlist)
# ----------------------------
def _to_date(x) -> Optional[dt.date]:
    if x is None:
        return None
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    # pandas Timestamp, numpy datetime64
    try:
        import pandas as pd  # type: ignore
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime().date()
    except Exception:
        pass
    try:
        # last resort: parse string
        return dt.datetime.fromisoformat(str(x)[:19]).date()
    except Exception:
        return None


def get_watchlist_earnings_next_days(tickers: List[str], days: int = 14) -> "pd.DataFrame":
    """
    Returns a dataframe of upcoming earnings dates for the supplied tickers within the next `days`.
    Best-effort using yfinance; if a ticker has no upcoming date, it is omitted.
    """
    import pandas as pd  # local import for faster CLI startup
    import yfinance as yf  # type: ignore

    today = dt.date.today()
    end_date = today + dt.timedelta(days=days)

    rows = []
    for tkr in tickers:
        try:
            yt = yf.Ticker(tkr)
            next_date: Optional[dt.date] = None

            # Preferred: earnings dates dataframe
            if hasattr(yt, "get_earnings_dates"):
                try:
                    df = yt.get_earnings_dates(limit=8)
                    if df is not None and len(df) > 0:
                        # index is Timestamp; pick first future one
                        for idx in df.index:
                            d = _to_date(idx)
                            if d and d >= today:
                                next_date = d
                                break
                except Exception:
                    pass

            # Fallback: calendar
            if next_date is None:
                try:
                    cal = getattr(yt, "calendar", None)
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if isinstance(ed, (list, tuple)) and ed:
                            next_date = _to_date(ed[0])
                        else:
                            next_date = _to_date(ed)
                except Exception:
                    pass

            if next_date and (today <= next_date <= end_date):
                rows.append({
                    "Ticker": tkr,
                    "Earnings Date": next_date.isoformat(),
                    "Days": (next_date - today).days,
                })
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out = df_out.sort_values(["Days", "Ticker"]).reset_index(drop=True)
    return df_out


def earnings_section_md(watchlist: List[str], days: int = 14) -> str:
    """
    Markdown section for upcoming earnings for watchlist tickers.
    """
    try:
        import pandas as pd  # type: ignore
        df = get_watchlist_earnings_next_days(watchlist, days=days)
        if df is None or df.empty:
            return f"## 3) Earnings next {days} days (watchlist)\n\n_None from watchlist in the next {days} days._\n"

        # Render as markdown table (right-align numeric)
        md = []
        md.append(f"## 3) Earnings next {days} days (watchlist)\n")
        md.append("_Upcoming earnings dates for your 44-ticker watchlist._\n")
        md.append(md_table_from_df(df, cols=["Ticker", "Earnings Date", "Days"]))
        return "\n".join(md) + "\n"
    except Exception:
        return f"## 3) Earnings next {days} days (watchlist)\n\n_(Failed to fetch earnings calendar.)_\n"



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
# Charting (signals)
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

    out = d[cols]

    # Alignment: textual columns left, numeric-ish columns right
    left_cols = {"Ticker", "Signal", "Pattern", "Dir", "Chart", "Instrument", "Symbol", "symbol"}
    aligns = tuple("left" if c in left_cols else "right" for c in cols)

    return df_to_markdown_aligned(out, aligns=aligns, index=False)


def diff_new_ended(prev: Dict[str, List[str]], cur: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    prev_set = set(prev.get("signals", []))
    cur_set = set(cur.get("signals", []))
    return sorted(cur_set - prev_set), sorted(prev_set - cur_set)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "custom"], default=os.environ.get("MODE", "custom"))
    ap.add_argument("--max-tickers", type=int, default=int(os.environ.get("MAX_TICKERS", "0")))
    args = ap.parse_args()

    custom = get_custom_tickers()

    # Universe for technical scan
    if args.mode == "full":
        spx = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        universe = sorted(set(custom + spx + ndx))
    else:
        universe = custom

    if args.max_tickers and args.max_tickers > 0:
        universe = universe[:args.max_tickers]

    now = dt.datetime.now(dt.timezone.utc)
    header_time = now.astimezone().strftime("%Y-%m-%d %H:%M %Z")

    # RSS
    rss_items = fetch_rss_headlines(limit_total=14)

    # 1) Snapshot table + exec summary
    snapshot_df = fetch_market_snapshot_multi()

    # 1) Macro "card" charts (5Y)
    vix_card = plot_gf_card_5y(
        "^VIX",
        "CBOE Volatility Index",
        "INDEXCBOE: VIX",
        "macro_vix_5y.png",
        decimals_last=2,
        line_color="#d93025",
    )
    eur_card = plot_gf_card_5y(
        "EURUSD=X",
        "Euro / US Dollar",
        "CCY: EURUSD",
        "macro_eurusd_5y.png",
        decimals_last=2,
        line_color="#d93025",
    )

    # Download OHLCV once (for technicals)
    ohlcv = yf_download_chunk(universe)

    # 2) Movers
    session_g, session_l = fetch_session_movers_yahoo()
    session_gf = filter_movers(session_g)
    session_lf = filter_movers(session_l)
    if not session_lf.empty:
        session_lf = session_lf.sort_values("pct", ascending=True)

    ah_g, ah_l = fetch_afterhours_movers()
    ah_gf = filter_movers(ah_g)
    ah_lf = filter_movers(ah_l)

    # Watchlist movers (>|4%|, incl. after-hours) for executive summary
    wl_set = set(universe)
    def _wl_extract(df: pd.DataFrame) -> List[Tuple[str, float]]:
        if df is None or df.empty:
            return []
        d = df.copy()
        d = d[d["symbol"].astype(str).isin(wl_set)]
        if d.empty:
            return []
        d = d.assign(abs_pct=d["pct"].abs()).sort_values("abs_pct", ascending=False)
        out = []
        for _, r in d.head(6).iterrows():
            try:
                out.append((str(r["symbol"]), float(r["pct"])))
            except Exception:
                continue
        return out

    # session_gf/session_lf already filtered to >= 4% absolute movers
    session_combined = pd.concat([session_gf, session_lf], ignore_index=True) if (session_gf is not None and session_lf is not None) else pd.DataFrame(columns=["symbol","pct"])
    ah_combined = pd.concat([ah_gf, ah_lf], ignore_index=True) if (ah_gf is not None and ah_lf is not None) else pd.DataFrame(columns=["symbol","pct"])

    watchlist_movers = {
        "session": _wl_extract(session_combined),
        "after_hours": _wl_extract(ah_combined),
    }

    if not ah_lf.empty:
        ah_lf = ah_lf.sort_values("pct", ascending=True)
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

    # Charts for signals
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

    # 1) Market recap & positioning (EXEC SUMMARY FIRST)
    md.append("## 1) Market recap & positioning\n")
    md.append("**Executive summary:**\n")
    exec_summary = build_exec_summary(snapshot_df, rss_items, watchlist_movers)
    md.append(exec_summary)
    md.append("")
    md.append("**Key tape (multi-horizon):**\n")
    md.append(format_snapshot_table_multi(snapshot_df))
    md.append("")

    md.append("**Macro charts (5Y):**\n")
    # Render as HTML with explicit sizing so dashboard + email match
    W = 414
    if vix_card and eur_card:
        md.append(
            f"<table><tr>"
            f"<td style=\"padding-right:12px;\"><img src=\"{vix_card}\" width=\"{W}\" style=\"width:{W}px;max-width:{W}px;height:auto;\"></td>"
            f"<td><img src=\"{eur_card}\" width=\"{W}\" style=\"width:{W}px;max-width:{W}px;height:auto;\"></td>"
            f"</tr></table>\n"
        )
    elif vix_card:
        md.append(f"<img src=\"{vix_card}\" width=\"{W}\" style=\"width:{W}px;max-width:{W}px;height:auto;\">\n")
    elif eur_card:
        md.append(f"<img src=\"{eur_card}\" width=\"{W}\" style=\"width:{W}px;max-width:{W}px;height:auto;\">\n")
    md.append("")

    # 2) Movers
    md.append("## 2) Biggest movers (≥ 4%)\n")
    md.append(movers_table(session_gf, "Session gainers"))
    md.append(movers_table(session_lf, "Session losers"))
    md.append(movers_table(ah_gf, "After-hours gainers"))
    md.append(movers_table(ah_lf, "After-hours losers"))

    # 3) Earnings (watchlist)
    md.append(earnings_section_md(WATCHLIST_44, days=14))

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

    # 5) Catalysts
    md.append("## 5) Needle-moving catalysts (RSS digest)\n")
    md.append("_Linked digest for drill-down._\n")
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
        try:
            base_url = os.environ.get('PUBLIC_BASE_URL', '').strip()
            write_email_assets(dt.datetime.now(dt.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z'), 'Run crashed; see traceback in report.', fallback, base_url, {'session': [], 'after_hours': []}, [], [])
        except Exception:
            pass
        raise SystemExit(0)
