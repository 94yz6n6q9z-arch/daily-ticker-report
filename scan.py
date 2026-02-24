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

# Watchlist performance table (grouped)
from tools.watchlist_perf import build_watchlist_performance_section_md

# ----------------------------
# Default watchlist (user-defined)
# ----------------------------
# Ensures your full 44-ticker watchlist is ALWAYS included when MODE=custom.
# You can disable this by setting USE_DEFAULT_WATCHLIST=0 in the environment.
WATCHLIST_44: List[str] = ["MELI","ARM","QBTS","IONQ","HOOD","PLTR","SNPS","AVGO","CDNS","AMAT",
    "NFLX","LRCX","TSM","DASH","ISRG","MUV2.DE","PGR","CMG","ANF","DECK",
    "NU","UCG.MI","MC.PA","RMS.PA","VST","OKLO","SMR","CEG","LEU","CCJ",
    "000660.KS","NVDA","NVO","LLY","AMZN","GOOGL","AAPL","META","MSFT","ASML",
    "WMT","BYDDY","RRTL.DE","ARR",
    "NAT","INSW","TNK","FRO","MPC","PSX","VLO","MAU.PA","REP.MC","CVX"
]


# ----------------------------
# Watchlist categories (for Section 6)
# ----------------------------
WATCHLIST_GROUPS: Dict[str, List[str]] = {
    # EDA merged into this bucket
    "AI compute & semis (incl. EDA)": ["NVDA","ARM","AVGO","TSM","000660.KS","ASML","AMAT","LRCX","SNPS","CDNS"],
    # Treat AMZN as E-commerce platform (cluster with MELI)
    "Big Tech platforms": ["AMZN","MELI","GOOGL","META","AAPL","MSFT","NFLX"],
    "Consumer & retail (incl. luxury)": ["WMT","RRTL.DE","ANF","DECK","MC.PA","RMS.PA","CMG","DASH","BYDDY"],
    # MUV2 is insurance (cluster with PGR)
    "Fintech & financials": ["HOOD","NU","PGR","MUV2.DE","UCG.MI","ARR"],
    "Healthcare": ["ISRG","LLY","NVO"],
    "Energy & Nuclear": ["VST","CEG","CCJ","LEU","OKLO","SMR"],
    # Single quantum bucket (no sub-splitting)
    "Quantum": ["IONQ","QBTS"],
    "Venezuela Oil": ["NAT","INSW","TNK","FRO","MPC","PSX","VLO","CVX","REP.MC","MAU.PA"],
}

# One-level-deeper subsegments (max 4 per category), implemented as ticker tags (no extra tables).
# These tags are used in:
# - Watchlist performance table (ticker column)
# - "Emerging chart trends (so what)" GPT rewrite (keeps tags when citing tickers)
SEGMENT_TAGS: Dict[str, str] = {
    # AI compute & semis (incl. EDA) — 4 segments
    "NVDA": "Compute/IP", "ARM": "Compute/IP", "AVGO": "Compute/IP",
    "TSM": "Foundry/Mem", "000660.KS": "Foundry/Mem",
    "ASML": "Equipment", "AMAT": "Equipment", "LRCX": "Equipment",
    "SNPS": "EDA", "CDNS": "EDA",

    # Big Tech platforms — 4 segments (AMZN grouped with MELI)
    "AMZN": "E-comm", "MELI": "E-comm",
    "GOOGL": "Ads", "META": "Ads",
    "AAPL": "Ecosystem", "MSFT": "Ecosystem",
    "NFLX": "Media",

    # Consumer & retail — 4 segments
    "WMT": "Defensive", "RRTL.DE": "Defensive",
    "ANF": "Brands", "DECK": "Brands",
    "MC.PA": "Luxury", "RMS.PA": "Luxury",
    "CMG": "Services", "DASH": "Services", "BYDDY": "Services",

    # Fintech & financials — 4 segments
    "HOOD": "Brokerage",
    "NU": "Fintech",
    "PGR": "Insurance", "MUV2.DE": "Insurance",
    "UCG.MI": "Bank/Yield", "ARR": "Bank/Yield",

    # Healthcare — 2 segments (still <= 4)
    "ISRG": "Medtech",
    "LLY": "Pharma", "NVO": "Pharma",

    # Energy & Nuclear — 4 segments
    "VST": "Power", "CEG": "Power",
    "CCJ": "Uranium",
    "LEU": "FuelCycle",
    "OKLO": "SMR", "SMR": "SMR",

    # Quantum — single segment
    "IONQ": "Quantum", "QBTS": "Quantum",

    # Venezuela Oil — 4 segments (keep cluster order in tables)
    "NAT": "Tanker", "INSW": "Tanker", "TNK": "Tanker", "FRO": "Tanker",
    "MPC": "Refiner", "PSX": "Refiner", "VLO": "Refiner",
    "CVX": "Integrated", "REP.MC": "Integrated",
    "MAU.PA": "Upstream",
}

def _base_ticker(t: str) -> str:
    # Display ticker without exchange suffix (e.g., MC.PA -> MC, RRTL.DE -> RRTL)
    return t.split(".", 1)[0] if "." in t else t

# Ticker display labels: include segment tag when available, but hide exchange suffix.
TICKER_LABELS: Dict[str, str] = {t: f"{_base_ticker(t)} ({seg})" for t, seg in SEGMENT_TAGS.items()}

def display_ticker(t: str) -> str:
    # Use segment-tag label when available, otherwise strip exchange suffix.
    return TICKER_LABELS.get(t, _base_ticker(t))


# Segment order for clustering inside tables (rank 0..3 within each category)
SEGMENT_ORDER: Dict[str, List[str]] = {
    "AI compute & semis (incl. EDA)": ["Compute/IP", "Foundry/Mem", "Equipment", "EDA"],
    "Big Tech platforms": ["E-comm", "Ads", "Ecosystem", "Media"],
    "Consumer & retail (incl. luxury)": ["Defensive", "Brands", "Luxury", "Services"],
    "Fintech & financials": ["Brokerage", "Fintech", "Insurance", "Bank/Yield"],
    "Healthcare": ["Medtech", "Pharma"],
    "Energy & Nuclear": ["Power", "Uranium", "FuelCycle", "SMR"],
    "Quantum": ["Quantum"],
    "Venezuela Oil": ["Tanker", "Refiner", "Integrated", "Upstream"],
}

# Build per-ticker rank so watchlist performance table clusters segments (e.g., refiners together).
TICKER_SEGMENT_RANK: Dict[str, int] = {}
for _cat, _ticks in WATCHLIST_GROUPS.items():
    order = SEGMENT_ORDER.get(_cat, [])
    idx_map = {seg: i for i, seg in enumerate(order)}
    for _t in _ticks:
        seg = SEGMENT_TAGS.get(_t)
        if seg is None:
            continue
        TICKER_SEGMENT_RANK[_t] = idx_map.get(seg, 99)
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

VOL_CONFIRM_MULT = 1.25   # volume must be >= 1.25x AvgVol(20) for CONFIRMED
CLV_BREAKOUT_MIN = 0.70   # CLV in [-1..+1] must be >= +0.70 for breakout confirmation
CLV_BREAKDOWN_MAX = -0.70  # CLV in [-1..+1] must be <= -0.70 for breakdown confirmation

LOOKBACK_DAYS = 260
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
    if "Ticker" in df.columns:
        df = df.copy()
        df["Ticker"] = df["Ticker"].astype(str).map(display_ticker)
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



def fetch_rss_headlines(limit_total: int = 18, max_age_hours: int = 48) -> List[Dict[str, str]]:
    """Fetch a diversified set of free headline feeds for daily context.

    - Best-effort: failures return fewer items (never crash the report).
    - De-dupes by title.
    - Filters to recent items when pubDate is parseable.
    """
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone, timedelta

    feeds = [
        ("Financial Times", "https://www.ft.com/?format=rss"),
        ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
        ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("CNBC Markets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
        # Investing.com provides multiple RSS feeds under /rss/news_*.rss (some may be region-locked).
        ("Investing.com", "https://www.investing.com/rss/news_25.rss"),
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters Top News", "https://feeds.reuters.com/reuters/topNews"),
        ("MarketWatch Top Stories", "https://feeds.marketwatch.com/marketwatch/topstories"),
    ]

    all_items: List[Dict[str, str]] = []
    for name, url in feeds:
        all_items.extend(parse_rss(url, name, limit=12))

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

    # Filter by recency when possible
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)

    def _ts(it: Dict[str, str]) -> float:
        pub = (it.get("pubDate") or "").strip()
        if not pub:
            return 0.0
        try:
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0

    recent: List[Dict[str, str]] = []
    for it in uniq:
        ts = _ts(it)
        if ts == 0.0:
            # keep undated items (some feeds omit pubDate)
            recent.append(it)
        else:
            if datetime.fromtimestamp(ts, tz=timezone.utc) >= cutoff:
                recent.append(it)

    # Sort newest-first when possible (undated items fall to bottom)
    recent.sort(key=_ts, reverse=True)

    return recent[:limit_total]



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

    Fixes two common failure modes:
    1) Invalid/unsupported model name (e.g., custom env values) -> model fallback ladder.
    2) 400s due to optional fields (reasoning/text) -> retry with minimal request.

    Returns None on failure so deterministic fallback can run.
    """

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    preferred = (os.environ.get("OPENAI_MODEL", "") or "").strip()
    candidates = [m for m in [preferred, "gpt-5.2-pro", "gpt-5.2-thinking", "gpt-4.1", "gpt-4o"] if m]
    seen = set(); models = []
    for m in candidates:
        if m not in seen:
            models.append(m); seen.add(m)

    effort = (os.environ.get("OPENAI_REASONING_EFFORT", "medium") or "medium").strip()

    instructions = """You are an experienced Financial Times markets editor.

Task: Write the Executive summary for a daily market report.

Output EXACTLY 2 or 3 sentences (no bullets, no headings).
Format rule: the FIRST sentence must start with the provided THEME_PHRASE followed by a colon.

Hard rules:
A) Use ONLY the provided market data + the provided headlines; do not invent events, names, or catalysts.
B) The first sentence must anchor to the provided DOMINANT_HEADLINE (paraphrase; no quotes).
C) Include concrete figures in sentence 1: at least NDX 1D, S&P 1D, and VIX 1D.
D) Contextualize today inside the last 3–4 weeks as a narrative (continuation/reversal of the recent tape).
   - You MAY use 7D/1M stats only as brief supporting evidence (max ONE short parenthetical).
   - Do NOT write a horizon comparison sentence like “Over the past month vs three months …”.
E) Mention watchlist movers ≥4% on BOTH sides if present (up to 2 gainers + 2 losers). If none, say so.
F) Only mention oil/FX when justified by headlines or clear linkage; otherwise omit.

Style: crisp, specific, FT-like. No filler (“markets moved”), no hype, no jargon like “risk-on”.
"""

    def _extract_text(data: dict) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
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
            return " ".join(parts).strip()
        return ""

    def _call(model: str, minimal: bool) -> Optional[str]:
        body = {
            "model": model,
            "instructions": instructions,
            "input": payload_text,
            "temperature": 0.2,
            "max_output_tokens": 220,
        }
        if not minimal:
            body["reasoning"] = {"effort": effort}
            body["text"] = {"verbosity": "low"}

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
        try:
            with urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            out_text = _extract_text(data)
            out_text = re.sub(r"\s+", " ", out_text).strip()
            if not out_text:
                return None
            sentences = re.split(r"(?<=[\.\!\?])\s+", out_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 3:
                out_text = " ".join(sentences[:3]).strip()
            print(f"[openai] exec success model={model} minimal={minimal}")
            return out_text
        except Exception as e:
            print(f"[openai] exec model={model} minimal={minimal} failed: {e}")
            return None

    for m in models:
        out = _call(m, minimal=False)
        if out:
            return out
        out = _call(m, minimal=True)
        if out:
            return out

    return None



def _openai_responses_watchlist_pulse(payload_text: str) -> Optional[str]:
    """Call OpenAI Responses API to rewrite the 'Emerging chart trends' watchlist pulse."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-5.2-pro").strip() or "gpt-5.2-pro"
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "medium").strip() or "medium"
    instructions = """You are an experienced markets editor.

Task: Summarize the watchlist technical-signal mix (“Emerging chart trends / so what”).

Output 4–6 numbered bullets (e.g., “1.”, “2.”). No headings.
Rules:
- Use ONLY the provided category_stats facts; do not invent catalysts.
- Focus on what the signal mix implies (leadership, risk appetite, sector rotation).
- Mention 1–3 tickers per bullet with their segment tags (in parentheses) when provided.
- Keep each bullet to one sentence, crisp and action-oriented.
"""

    body = {
        "model": model,
        "instructions": instructions,
        "input": payload_text,
        "temperature": 0.3,
        "max_output_tokens": 260,
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
                    out_text = "\n".join(parts).strip()

        out_text = out_text.strip()
        if not out_text:
            return None

        lines = [ln.strip() for ln in out_text.splitlines() if ln.strip()]
        numbered = [ln for ln in lines if re.match(r"^\d+\.", ln)]
        if len(numbered) >= 4:
            return "\n".join(numbered[:6])
        return None
    except Exception:
        return None


def _sig_stage_weight(sig: str) -> int:
    if sig.startswith("CONFIRMED_"):
        return 3
    if sig.startswith("EARLY_"):
        return 1
    return 1


def _sig_direction(sig: str) -> int:
    if "BREAKOUT" in sig:
        return +1
    if "BREAKDOWN" in sig:
        return -1
    return 0


def _dominant_signal(signals: List[Tuple[str, float, bool]]) -> Optional[Tuple[str, float, bool]]:
    if not signals:
        return None
    ranked = []
    for s, dist, is_new in signals:
        ranked.append((_sig_stage_weight(s), 1 if is_new else 0, abs(dist), s, dist, is_new))
    ranked.sort(reverse=True)
    _, _, _, s, dist, is_new = ranked[0]
    return (s, dist, is_new)


def build_watchlist_pulse_section_md(
    df_early_new: pd.DataFrame,
    df_early_old: pd.DataFrame,
    df_trig_new: pd.DataFrame,
    df_trig_old: pd.DataFrame,
    watchlist_groups: Dict[str, List[str]],
    ticker_labels: Dict[str, str],
) -> str:
    def _iter_df(df: pd.DataFrame, is_new: bool) -> List[Tuple[str, str, float, bool]]:
        if df is None or df.empty:
            return []
        out = []
        for _, r in df.iterrows():
            t = str(r.get("Ticker", "")).strip()
            s = str(r.get("Signal", "")).strip()
            dist = r.get("Dist(ATR)", float("nan"))
            try:
                dist = float(dist)
            except Exception:
                dist = float("nan")
            out.append((t, s, dist, is_new))
        return out

    rows = []
    rows += _iter_df(df_early_new, True)
    rows += _iter_df(df_early_old, False)
    rows += _iter_df(df_trig_new, True)
    rows += _iter_df(df_trig_old, False)

    sigs_by_t: Dict[str, List[Tuple[str, float, bool]]] = {}
    for t, s, dist, is_new in rows:
        if not t or not s:
            continue
        sigs_by_t.setdefault(t, []).append((s, 0.0 if math.isnan(dist) else dist, is_new))

    cat_stats = {}
    for cat, tickers in watchlist_groups.items():
        counts = {"CONF_UP": 0, "CONF_DN": 0, "EARLY_UP": 0, "EARLY_DN": 0}
        score = 0
        leaders = []
        for t in tickers:
            dom = _dominant_signal(sigs_by_t.get(t, []))
            if not dom:
                continue
            sig, dist, is_new = dom
            w = _sig_stage_weight(sig)
            d = _sig_direction(sig)
            if d == 0:
                continue
            key = ("CONF" if w == 3 else "SOFT" if w == 2 else "EARLY") + ("_UP" if d > 0 else "_DN")
            counts[key] += 1
            score += w * d
            label = ticker_labels.get(t, t)
            leaders.append((w, 1 if is_new else 0, abs(dist), label, sig))
        leaders.sort(reverse=True)
        top = [{"ticker": x[3], "signal": x[4]} for x in leaders[:3]]
        cat_stats[cat] = {"score": score, "counts": counts, "top": top}

    md = []
    md.append("### 4) Emerging chart trends (the actual “so what”)")
    md.append("")
    md.append("_Logic: score each ticker by stage (CONFIRMED=3, EARLY=1) × direction (BREAKOUT=+1, BREAKDOWN=-1), then aggregate by category._")
    md.append("")
    md.append("| Category | Bias | CONF↑ | CONF↓ | EARLY↑ | EARLY↓ |")
    md.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for cat, s in cat_stats.items():
        sc = s["score"]
        bias = "Bullish" if sc >= 3 else "Bearish" if sc <= -3 else "Mixed"
        c = s["counts"]
        md.append(f"| {cat} | {bias} | {c['CONF_UP']} | {c['CONF_DN']} | {c['EARLY_UP']} | {c['EARLY_DN']} |")
    md.append("")

    payload = {
        "category_stats": [
            {"category": cat, "score": s["score"], **s["counts"], "top": s["top"]}
            for cat, s in cat_stats.items()
        ],
        "note": "Facts are derived from the watchlist technical triggers tables (early callouts + breakouts/breakdowns).",
    }

    llm = _openai_responses_watchlist_pulse(json.dumps(payload, ensure_ascii=False))
    if llm:
        md.append(llm)
    else:
        # deterministic fallback (short)
        cats_sorted = sorted(cat_stats.items(), key=lambda kv: kv[1]["score"])
        bearish = [kv for kv in cats_sorted if kv[1]["score"] <= -3]
        bullish = [kv for kv in reversed(cats_sorted) if kv[1]["score"] >= 3]
        bullets = []
        if bullish and bearish:
            bullets.append(f"1. **Leadership is split** — {bullish[0][0]} leads while {bearish[0][0]} leans bearish; reads as selective risk-taking rather than broad risk-on.")
        n = 2
        for cat, s in bullish[:2]:
            tops = ", ".join([f"{x['ticker']} ({x['signal']})" for x in s["top"]]) or "—"
            bullets.append(f"{n}. **{cat} is a tailwind** — signals skew bullish. Key names: {tops}.")
            n += 1
        for cat, s in bearish[:2]:
            tops = ", ".join([f"{x['ticker']} ({x['signal']})" for x in s["top"]]) or "—"
            bullets.append(f"{n}. **{cat} is a headwind** — breakdowns dominate. Key names: {tops}.")
            n += 1
        md.extend(bullets[:6])

    md.append("")
    return "\n".join(md)


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

    Prefer GPT prose via OpenAI API; fall back to deterministic text if API missing/fails.
    """
    if snapshot_df is None or snapshot_df.empty:
        return "Market summary unavailable (snapshot empty)."

    def row(name: str) -> Optional[pd.Series]:
        x = snapshot_df.loc[snapshot_df["Instrument"] == name]
        return None if x.empty else x.iloc[0]

    ndx = row("Nasdaq 100")
    spx = row("S&P 500")
    vix = row("VIX")
    wti = row("WTI Crude")
    eur = row("EUR/USD")
    stx = row("STOXX Europe 600")
    dax = row("DAX")

    def f(r: Optional[pd.Series], key: str) -> float:
        try:
            if r is None:
                return float("nan")
            return float(r.get(key, np.nan))
        except Exception:
            return float("nan")

    # Headline payload (source + title + link) so the model can anchor the story.
    top_headlines = [
        {
            "source": (it.get("source", "") or "").strip(),
            "title": (it.get("title", "") or "").strip(),
            "link": (it.get("link", "") or "").strip(),
        }
        for it in (rss_items or [])
        if (it.get("title") or "").strip()
    ][:12]


    def pick_dominant(items: List[dict]) -> Optional[dict]:
        if not items:
            return None

        def score(it: dict) -> int:
            title = (it.get("title", "") or "").lower()
            src = (it.get("source", "") or "").lower()
            s = 0
            if "cnbc" in src:
                s += 3
            if "yahoo" in src:
                s += 2
            if "invest" in src:
                s += 2
            if "reuters" in src:
                s += 4
            for kw in [
                "fed",
                "rates",
                "inflation",
                "cpi",
                "powell",
                "tariff",
                "supreme court",
                "iran",
                "ukraine",
                "gaza",
                "earnings",
                "guidance",
                "ai",
                "nvidia",
                "stocks",
                "market",
            ]:
                if kw in title:
                    s += 2
            return s

        return max(items, key=score)

    dominant = pick_dominant(top_headlines)
    theme_phrase = "Market wrap"
    if dominant and (dominant.get("title") or "").strip():
        dom_title = str(dominant.get("title", "")).strip()
        words = re.sub(r"[^A-Za-z0-9\s\-]", "", dom_title).split()
        theme_phrase = " ".join(words[:8]).strip() or "Market wrap"

    payload = {
        "market": {
            "NDX": {"1D": f(ndx, "1D"), "7D": f(ndx, "7D"), "1M": f(ndx, "1M")},
            "S&P": {"1D": f(spx, "1D"), "7D": f(spx, "7D"), "1M": f(spx, "1M")},
            "VIX": {"1D": f(vix, "1D"), "7D": f(vix, "7D"), "1M": f(vix, "1M")},
            "WTI": {"1D": f(wti, "1D"), "7D": f(wti, "7D"), "1M": f(wti, "1M")} if wti is not None else None,
            "EURUSD": {"1D": f(eur, "1D")} if eur is not None else None,
            "STOXX": {"1D": f(stx, "1D")} if stx is not None else None,
            "DAX": {"1D": f(dax, "1D")} if dax is not None else None,
        },
        "watchlist_movers_over_4pct": {
            "session": watchlist_movers.get("session", []),
            "after_hours": watchlist_movers.get("after_hours", []),
        },
        "headline_themes": summarize_rss_themes(rss_items),
        "dominant_headline": dominant,
        "theme_phrase": theme_phrase,
        "headlines": top_headlines,
    }

    llm = _openai_responses_exec_summary(json.dumps(payload, ensure_ascii=False))
    if llm:
        # Enforce the theme opener so the first sentence always starts with the dominant driver phrase.
        try:
            tp = (theme_phrase or "").strip()
            if tp:
                low = llm.strip().lower()
                want = tp.lower() + ':'
                if not low.startswith(want):
                    # If the model already used another opener, keep the body but replace the opener.
                    llm = tp + ': ' + llm.strip()
        except Exception:
            pass
        return llm

    # Deterministic fallback (only used if OpenAI API is missing/fails)
    # Use the dominant headline to anchor the first sentence, and show both gainers/losers.
    dom_hint = ""
    if dominant and (dominant.get("title") or "").strip():
        dom_hint = f" (per {str(dominant.get('source','')).strip()} — {str(dominant.get('title','')).strip()})"

    movers = watchlist_movers.get("session", []) + watchlist_movers.get("after_hours", [])
    gainers = sorted([x for x in movers if x[1] >= MOVER_THRESHOLD_PCT], key=lambda z: z[1], reverse=True)
    losers = sorted([x for x in movers if x[1] <= -MOVER_THRESHOLD_PCT], key=lambda z: z[1])

    if not gainers and not losers:
        movers_txt = "No watchlist names moved >4% incl. after-hours."
    else:
        def _fmt(items):
            return ", ".join([f"{t} {p:+.1f}%" for t, p in items])
        g = _fmt(gainers[:2])
        l = _fmt(losers[:2])
        movers_txt = "Watchlist movers >4%: " + " | ".join([x for x in [g, l] if x]) + "."

    s1 = (
        f"{theme_phrase}: NDX {f(ndx,'1D'):+.2f}% vs S&P {f(spx,'1D'):+.2f}%, with VIX {f(vix,'1D'):+.2f}% as investors digested the day’s lead story{dom_hint}."
    )

    # Narrative context (avoid horizon comparisons)
    s2 = (
        "The move landed in a market that’s been choppy recently, with investors oscillating between macro/rates and growth/AI positioning."
    )

    return s1 + " " + s2 + " " + movers_txt


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
    t["Ticker"] = t["symbol"].astype(str).map(display_ticker)
    t["pct"] = pd.to_numeric(t["pct"], errors="coerce").map(lambda x: f"{x:+.2f}%")
    out = t[["Ticker", "pct"]]
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


def _classify_vs_level(
    close: float,
    level: float,
    atr_val: float,
    direction: str,
    vol_ratio: float,
    clv: float,
) -> Tuple[str, float]:
    """Classify signal strength vs a trigger level with hard confirmation gates.

    CONFIRMED requires ALL:
      1) close beyond trigger by >= ATR_CONFIRM_MULT * ATR(14)
      2) volume ratio >= VOL_CONFIRM_MULT (vs AvgVol(20))
      3) CLV >= CLV_BREAKOUT_MIN for breakouts, <= CLV_BREAKDOWN_MAX for breakdowns (CLV in [-1..+1])

    EARLY is within EARLY_MULT * ATR of the trigger (pre-plan zone); if not CONFIRMED, we keep it EARLY (no SOFT tier).
    """
    if atr_val is None or atr_val <= 0 or math.isnan(atr_val):
        atr_val = max(level * 0.01, 1e-6)

    if vol_ratio is None or (isinstance(vol_ratio, float) and math.isnan(vol_ratio)):
        vol_ratio = 1.0
    if clv is None or (isinstance(clv, float) and math.isnan(clv)):
        clv = 0.0

    dist_atr = (close - level) / atr_val

    if direction == "BREAKOUT":
        price_ok = close >= level + ATR_CONFIRM_MULT * atr_val
        vol_ok = vol_ratio >= VOL_CONFIRM_MULT
        clv_ok = clv >= CLV_BREAKOUT_MIN
        if price_ok and vol_ok and clv_ok:
            return "CONFIRMED_", dist_atr
        # No SOFT tier: if not confirmed, keep only EARLY when close is within the pre-plan zone.
        if abs(close - level) <= EARLY_MULT * atr_val:
            return "EARLY_", dist_atr
        return "", dist_atr

    # BREAKDOWN
    price_ok = close <= level - ATR_CONFIRM_MULT * atr_val
    vol_ok = vol_ratio >= VOL_CONFIRM_MULT
    clv_ok = clv <= CLV_BREAKDOWN_MAX
    if price_ok and vol_ok and clv_ok:
        return "CONFIRMED_", dist_atr
    # No SOFT tier: if not confirmed, keep only EARLY when close is within the pre-plan zone.
    if abs(close - level) <= EARLY_MULT * atr_val:
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

    # Confirmation gates use volume ratio (vs AvgVol20) and CLV ([-1..+1])
    vol_ratio = 1.0
    if "Volume" in d.columns and not d["Volume"].dropna().empty:
        v = float(d["Volume"].iloc[-1]) if not math.isnan(float(d["Volume"].iloc[-1])) else float("nan")
        avg20 = float(d["Volume"].tail(20).mean()) if len(d) >= 20 else float("nan")
        if avg20 and not math.isnan(avg20) and not math.isnan(v):
            vol_ratio = v / avg20

    clv = 0.0
    try:
        hi = float(d["High"].iloc[-1])
        lo = float(d["Low"].iloc[-1])
        if hi > lo:
            clv = (2.0 * close - hi - lo) / (hi - lo)  # CLV in [-1..+1]
            clv = max(-1.0, min(1.0, float(clv)))
    except Exception:
        clv = 0.0

    hs = detect_hs_top(d)
    if hs:
        pattern, direction, level = hs
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction, vol_ratio, clv)
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
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction, vol_ratio, clv)
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

        prefix_u, dist_u = _classify_vs_level(close, upper_level, atr_val, "BREAKOUT", vol_ratio, clv)
        if prefix_u:
            sigs.append(LevelSignal(
                ticker=ticker, signal=f"{prefix_u}{pattern}_BREAKOUT",
                pattern=pattern, direction="BREAKOUT",
                level=float(upper_level), close=close, atr=atr_val,
                dist_atr=float(dist_u), pct_today=pct_today
            ))

        prefix_l, dist_l = _classify_vs_level(close, lower_level, atr_val, "BREAKDOWN", vol_ratio, clv)
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
def _pivots(arr: np.ndarray, w: int = 5, kind: str = "high") -> List[int]:
    piv: List[int] = []
    for i in range(w, len(arr) - w):
        window = arr[i - w : i + w + 1]
        if kind == "high":
            if arr[i] == np.max(window) and np.sum(window == arr[i]) == 1:
                piv.append(i)
        else:
            if arr[i] == np.min(window) and np.sum(window == arr[i]) == 1:
                piv.append(i)
    return piv

def pivots(arr: np.ndarray, w: int = 5, kind: str = "high") -> List[int]:
    """Alias for _pivots (backward-compatible)."""
    return _pivots(arr, w=w, kind=kind)



def _annotate_hs_top(ax, close: np.ndarray, low: np.ndarray) -> None:
    piv = _pivots(close, w=5, kind="high")[-12:]
    if len(piv) < 3:
        return
    best = None
    for i in range(len(piv) - 2):
        a, b, c = piv[i], piv[i + 1], piv[i + 2]
        if close[b] > close[a] and close[b] > close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)
    if best is None:
        best = (piv[-3], piv[-2], piv[-1])
    ls, head, rs = best
    for idx, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([idx], [close[idx]], s=40)
        ax.annotate(label, (idx, close[idx]), xytext=(idx, close[idx] + 3),
                    arrowprops=dict(arrowstyle="->", lw=1))
    n1 = float(np.min(low[min(ls, head) : max(ls, head) + 1]))
    n2 = float(np.min(low[min(head, rs) : max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0
    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(len(close) - 1, neckline, " Neckline", va="bottom")


def _annotate_ihs(ax, close: np.ndarray, high: np.ndarray) -> None:
    piv = _pivots(close, w=5, kind="low")[-12:]
    if len(piv) < 3:
        return
    best = None
    for i in range(len(piv) - 2):
        a, b, c = piv[i], piv[i + 1], piv[i + 2]
        if close[b] < close[a] and close[b] < close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)
    if best is None:
        best = (piv[-3], piv[-2], piv[-1])
    ls, head, rs = best
    for idx, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([idx], [close[idx]], s=40)
        ax.annotate(label, (idx, close[idx]), xytext=(idx, close[idx] - 4),
                    arrowprops=dict(arrowstyle="->", lw=1))
    n1 = float(np.max(high[min(ls, head) : max(ls, head) + 1]))
    n2 = float(np.max(high[min(head, rs) : max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0
    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(len(close) - 1, neckline, " Neckline", va="bottom")






def _annotate_hs_top_dt(ax, dates, close, low) -> Optional[float]:
    """Date-aware HS-top labeling (avoids date-axis distortion)."""
    piv_hi = pivots(close, w=5, kind="high")[-10:]
    if len(piv_hi) < 3:
        return None
    best = None
    for i in range(len(piv_hi) - 2):
        a, b, c = piv_hi[i], piv_hi[i + 1], piv_hi[i + 2]
        if close[b] > close[a] and close[b] > close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)
    if not best:
        best = (piv_hi[-3], piv_hi[-2], piv_hi[-1])
    ls, head, rs = best

    for idxp, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([dates[idxp]], [close[idxp]], s=40)
        ax.annotate(label, (dates[idxp], close[idxp]),
                    xytext=(dates[idxp], close[idxp] + 3),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", lw=1))

    n1 = float(np.min(low[min(ls, head):max(ls, head) + 1]))
    n2 = float(np.min(low[min(head, rs):max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0
    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(dates[-1], neckline, " Neckline", va="bottom")
    return neckline


def _annotate_ihs_dt(ax, dates, close, high) -> Optional[float]:
    """Date-aware IHS labeling (avoids date-axis distortion)."""
    piv_lo = pivots(close, w=5, kind="low")[-10:]
    if len(piv_lo) < 3:
        return None
    best = None
    for i in range(len(piv_lo) - 2):
        a, b, c = piv_lo[i], piv_lo[i + 1], piv_lo[i + 2]
        if close[b] < close[a] and close[b] < close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)
    if not best:
        best = (piv_lo[-3], piv_lo[-2], piv_lo[-1])
    ls, head, rs = best

    for idxp, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([dates[idxp]], [close[idxp]], s=40)
        ax.annotate(label, (dates[idxp], close[idxp]),
                    xytext=(dates[idxp], close[idxp] - 4),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", lw=1))

    n1 = float(np.max(high[min(ls, head):max(ls, head) + 1]))
    n2 = float(np.max(high[min(head, rs):max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0
    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(dates[-1], neckline, " Neckline", va="bottom")
    return neckline
def _annotate_wedge(ax, dates, high, low, lookback: int = 120) -> None:
    """
    Best-effort wedge visual:
    - fit upper trendline through pivot highs
    - fit lower trendline through pivot lows
    - scatter pivot points (touches) used for the fit
    Works for both WEDGE_UP_* and WEDGE_DOWN_*.
    """
    import numpy as _np

    n = len(high)
    if n < 40:
        return
    lb = min(lookback, n)
    hi = _np.asarray(high[-lb:], dtype=float)
    lo = _np.asarray(low[-lb:], dtype=float)
    dts = dates[-lb:]

    def pivots(arr, w=4, kind="high"):
        out = []
        for i in range(w, len(arr)-w):
            win = arr[i-w:i+w+1]
            if kind == "high":
                if arr[i] == _np.max(win):
                    out.append(i)
            else:
                if arr[i] == _np.min(win):
                    out.append(i)
        return out

    piv_hi = pivots(hi, w=4, kind="high")[-4:]
    piv_lo = pivots(lo, w=4, kind="low")[-4:]
    if len(piv_hi) < 2 or len(piv_lo) < 2:
        return

    xh = _np.array(piv_hi, dtype=float)
    yh = hi[piv_hi]
    xl = _np.array(piv_lo, dtype=float)
    yl = lo[piv_lo]

    # Fit lines y = a*x + b
    ah, bh = _np.polyfit(xh, yh, 1)
    al, bl = _np.polyfit(xl, yl, 1)

    xs = _np.arange(lb, dtype=float)
    upper = ah*xs + bh
    lower = al*xs + bl

    # plot lines
    ax.plot(dts, upper, linestyle="--", linewidth=1)
    ax.plot(dts, lower, linestyle="--", linewidth=1)
    # touches
    ax.scatter([dts[i] for i in piv_hi], yh, s=22)
    ax.scatter([dts[i] for i in piv_lo], yl, s=22)

    # label
    ax.text(dts[int(lb*0.02)], upper[int(lb*0.05)], "Wedge upper", fontsize=9)
    ax.text(dts[int(lb*0.02)], lower[int(lb*0.10)], "Wedge lower", fontsize=9)
def plot_signal_chart(ticker: str, df: pd.DataFrame, sig: LevelSignal) -> Optional[str]:
    """
    Chart output (last ~1Y, with indicators):
      - Close (line)
      - SMA(50) and SMA(200)
      - Volume subplot
      - Trigger (sig.level) + Confirm (±0.5 ATR)
      - Pattern markings:
          * HS/IHS: LS/H/RS + neckline
          * WEDGE: upper/lower lines + "touch" pivots
    Always returns a chart path; if anything fails, writes a placeholder PNG.
    """
    fname = f"{ticker}_{sig.signal}.png"
    fname = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", fname)
    out_path = IMG_DIR / fname
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    def placeholder(reason: str) -> str:
        fig = plt.figure(figsize=(10.5, 5.0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.75, f"{display_ticker(ticker)}", fontsize=16, weight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.58, f"{sig.signal}", fontsize=12, transform=ax.transAxes)
        ax.text(0.02, 0.40, "Chart unavailable", fontsize=12, transform=ax.transAxes)
        ax.text(0.02, 0.25, f"Reason: {reason}", fontsize=10, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return f"img/{fname}"

    if df is None or df.empty:
        return placeholder("no data")

    # --- Clean + ensure datetime index ---
    d0 = df.copy()
    # keep needed
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in d0.columns:
            if col == "Volume":
                d0[col] = np.nan
            else:
                return placeholder(f"missing column {col}")

    # Drop rows with invalid OHLC (common on weekends/partials for some tickers)
    d0 = d0.dropna(subset=["Close", "High", "Low"]).copy()
    if d0.empty or len(d0) < 80:
        return placeholder("insufficient history")

    try:
        # Ensure datetime index; avoid accidental epoch (1970) axes
        if not isinstance(d0.index, pd.DatetimeIndex):
            # If there's a Date column, use it; otherwise synthesize business-day index
            if "Date" in d0.columns:
                d0["Date"] = pd.to_datetime(d0["Date"], errors="coerce")
                d0 = d0.dropna(subset=["Date"]).set_index("Date")
            else:
                d0.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(d0))
        else:
            # Clean any non-datetime artifacts
            idx = pd.to_datetime(d0.index, errors="coerce")
            d0 = d0.loc[~idx.isna()].copy()
            d0.index = pd.to_datetime(d0.index, errors="coerce")

        if d0.empty:
            return placeholder("could not parse dates")

        d0 = d0.sort_index()

        # Guard against epoch/outlier dates (e.g., 1970) by using last 400 rows then date-filter
        d_full = d0.tail(420).copy()

        # Plot window = last ~1 year
        last_dt = d_full.index.max()
        cutoff = last_dt - pd.Timedelta(days=370)
        d = d_full.loc[d_full.index >= cutoff].copy()
        if len(d) < 80:
            d = d_full.tail(260).copy()

        # Indicators (computed on d_full so SMA200 works)
        sma50_full = d_full["Close"].rolling(50).mean()
        sma200_full = d_full["Close"].rolling(200).mean()
        sma50 = sma50_full.loc[d.index]
        sma200 = sma200_full.loc[d.index]

        # ATR(14)
        atr_s = atr(d_full, 14)
        atr_last = float(atr_s.dropna().iloc[-1]) if atr_s is not None and len(atr_s.dropna()) else 0.0

        # Confirm line per rule
        direction = 1 if "BREAKOUT" in sig.signal else -1 if "BREAKDOWN" in sig.signal else 0
        confirm = sig.level + direction * 0.5 * atr_last

        # --- Build figure with volume subplot ---
        fig, (ax, axv) = plt.subplots(
            2, 1,
            figsize=(10.8, 6.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0]}
        )

        # Price + SMAs
        ax.plot(d.index, d["Close"].astype(float).values)
        ax.plot(d.index, sma50.astype(float).values)
        ax.plot(d.index, sma200.astype(float).values)

        # Trigger + confirm
        ax.axhline(sig.level, linestyle="-.", linewidth=1)
        ax.axhline(confirm, linestyle=":", linewidth=1)

        ax.text(d.index[-1], sig.level, " Trigger", va="bottom")
        ax.text(d.index[-1], confirm, " Confirm (±0.5 ATR)", va="bottom")

        # Pattern markings
        close = d["Close"].astype(float).values
        high = d["High"].astype(float).values
        low = d["Low"].astype(float).values

        if "HS_TOP" in sig.signal or "H&S_TOP" in sig.signal:
            _annotate_hs_top_dt(ax, d.index.to_list(), close, low)
            # Label neckline (visual anchor)
            ax.text(d.index[int(len(d)*0.05)], sig.level, "Neckline", va="bottom")
        if "IHS" in sig.signal:
            _annotate_ihs_dt(ax, d.index.to_list(), close, high)
            ax.text(d.index[int(len(d)*0.05)], sig.level, "Neckline", va="bottom")
        if "WEDGE" in sig.signal:
            _annotate_wedge(ax, d.index.to_list(), high, low, lookback=min(140, len(d)))

        # Latest close marker
        ax.scatter([d.index[-1]], [close[-1]], s=60)
        ax.annotate("Close", (d.index[-1], close[-1]),
                    xytext=(d.index[-1], close[-1]),
                    textcoords="data")

        # Trade-prep box
        box = f"Trigger: {sig.level:.2f}\\nConfirm: {confirm:.2f}\\nDist: {sig.dist_atr:+.2f} ATR"
        ax.text(0.02, 0.02, box, transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.6))

        # Volume subplot
        vol = d["Volume"].fillna(0).astype(float).values
        axv.bar(d.index, vol, width=1.0)
        axv.set_ylabel("Vol")

        title = f"{display_ticker(ticker)} | {sig.signal}"
        ax.set_title(title)
        ax.set_ylabel("Close")
        axv.set_xlabel("Date")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return f"img/{fname}"

    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        return placeholder(str(e))


# ----------------------------

def blurb_for_new_signal(sig: LevelSignal) -> str:
    """
    Short explanation for NEW early callouts (used in 4A).
    Kept deterministic (no macro storytelling).
    """
    direction = "breakout" if "BREAKOUT" in sig.signal else "breakdown" if "BREAKDOWN" in sig.signal else "move"
    pattern = sig.pattern if sig.pattern else "pattern"
    lines = []
    lines.append(f"**{display_ticker(sig.ticker)} — {sig.signal}**")
    lines.append(f"- **Pattern:** {pattern} ({direction}).")
    lines.append(f"- **Trigger (level):** {sig.level:.2f} | **Distance:** {sig.dist_atr:+.2f} ATR.")
    lines.append(f"- **Plan:** wait for a close beyond trigger by ≥ 0.5 ATR (confirmation) or a clean retest/failure depending on direction.")
    if "WEDGE" in sig.signal:
        lines.append("- **Wedge visual:** chart shows upper/lower trendlines with recent touch points; trigger is drawn at the breakout boundary.")
    if "HS_TOP" in sig.signal or "IHS" in sig.signal:
        lines.append("- **HS/IHS visual:** chart labels LS/H/RS and the neckline; trigger is the neckline.")
    return "\n".join(lines)
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
    if "Vol/AvgVol20" in d.columns:
        d["Vol/AvgVol20"] = pd.to_numeric(d["Vol/AvgVol20"], errors="coerce").map(lambda x: f"{x:.2f}×" if pd.notna(x) else "")
    if "CLV" in d.columns:
        d["CLV"] = pd.to_numeric(d["CLV"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    if "Day%" in d.columns:
        d["Day%"] = pd.to_numeric(d["Day%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "Chart" in d.columns:
        d["Chart"] = d["Chart"].apply(lambda p: f"[chart]({p})" if isinstance(p, str) and p else "")

    out = d[cols]

    # Alignment: textual columns left, numeric-ish columns right
    left_cols = {"Ticker", "Signal", "Pattern", "Dir", "Chart", "Instrument", "Symbol", "symbol"}
    aligns = tuple("left" if c in left_cols else "right" for c in cols)

    return df_to_markdown_aligned(out, aligns=aligns, index=False)


def enrich_confirmed_rules(df: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add confirmation-gate diagnostics to CONFIRMED tables.

    Adds:
      - Threshold (same as Level)
      - Vol/AvgVol20
      - CLV ([-1..+1])
      - PriceOK / VolOK / CLVOK (✅/❌)
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    if "Threshold" not in d.columns:
        d["Threshold"] = d["Level"] if "Level" in d.columns else np.nan

    vols = []
    clvs = []
    price_ok = []
    vol_ok = []
    clv_ok = []

    for _, r in d.iterrows():
        t = str(r.get("Ticker", "")).strip()
        sig = str(r.get("Signal", ""))
        dist = pd.to_numeric(r.get("Dist(ATR)", np.nan), errors="coerce")
        is_breakout = "BREAKOUT" in sig and "BREAKDOWN" not in sig
        is_breakdown = "BREAKDOWN" in sig

        # Price gate from Dist(ATR)
        p_ok = False
        if pd.notna(dist):
            if is_breakout:
                p_ok = dist >= ATR_CONFIRM_MULT
            elif is_breakdown:
                p_ok = dist <= -ATR_CONFIRM_MULT
        # Vol/CLV from OHLCV
        vr = np.nan
        cv = np.nan
        td = ohlcv.get(t)
        if td is not None and not td.empty:
            td2 = td.dropna(subset=["Open","High","Low","Close"])
            if not td2.empty:
                try:
                    close = float(td2["Close"].iloc[-1])
                    hi = float(td2["High"].iloc[-1])
                    lo = float(td2["Low"].iloc[-1])
                    if hi > lo:
                        cv = (2.0*close - hi - lo) / (hi - lo)
                        cv = max(-1.0, min(1.0, float(cv)))
                except Exception:
                    cv = np.nan
                if "Volume" in td2.columns and not td2["Volume"].dropna().empty and len(td2) >= 2:
                    try:
                        v = float(td2["Volume"].iloc[-1])
                        avg20 = float(td2["Volume"].tail(20).mean()) if len(td2) >= 20 else np.nan
                        if avg20 and not math.isnan(avg20) and not math.isnan(v):
                            vr = v / avg20
                    except Exception:
                        vr = np.nan

        v_ok = bool(pd.notna(vr) and vr >= VOL_CONFIRM_MULT)
        c_ok = False
        if pd.notna(cv):
            if is_breakout:
                c_ok = cv >= CLV_BREAKOUT_MIN
            elif is_breakdown:
                c_ok = cv <= CLV_BREAKDOWN_MAX

        vols.append(vr)
        clvs.append(cv)
        price_ok.append("✅" if p_ok else "❌")
        vol_ok.append("✅" if v_ok else "❌")
        clv_ok.append("✅" if c_ok else "❌")

    d["Vol/AvgVol20"] = vols
    d["CLV"] = clvs
    d["PriceOK"] = price_ok
    d["VolOK"] = vol_ok
    d["CLVOK"] = clv_ok
    return d


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
    # Compute session movers from our watchlist universe (more reliable than scraping Yahoo gainers/losers)
    session_rows = []
    for t in universe:
        d = ohlcv.get(t)
        if d is None or d.empty or "Close" not in d.columns:
            continue
        dd = d.dropna(subset=["Close"])
        if len(dd) < 2:
            continue
        c0 = float(dd["Close"].iloc[-2])
        c1 = float(dd["Close"].iloc[-1])
        if c0 == 0 or math.isnan(c0) or math.isnan(c1):
            continue
        pct = (c1 / c0 - 1.0) * 100.0
        session_rows.append({"symbol": t, "pct": float(pct)})
    session_all = pd.DataFrame(session_rows, columns=["symbol", "pct"])
    session_gf = session_all[session_all["pct"] >= MOVER_THRESHOLD_PCT].sort_values("pct", ascending=False)
    session_lf = session_all[session_all["pct"] <= -MOVER_THRESHOLD_PCT].sort_values("pct", ascending=True)

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
        d["pct"] = pd.to_numeric(d["pct"], errors="coerce")
        d = d.dropna(subset=["pct"])
        g = d[d["pct"] >= MOVER_THRESHOLD_PCT].sort_values("pct", ascending=False).head(3)
        l = d[d["pct"] <= -MOVER_THRESHOLD_PCT].sort_values("pct", ascending=True).head(3)
        out: List[Tuple[str, float]] = []
        for _, r in pd.concat([g, l], ignore_index=True).iterrows():
            out.append((str(r["symbol"]), float(r["pct"])))
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
    triggered = [s for s in all_signals if s.signal.startswith("CONFIRMED_")]

    def rank_trigger(s: LevelSignal) -> Tuple[int, float]:
        tier = 0
        return (tier, abs(s.dist_atr))

    triggered_sorted = sorted(triggered, key=rank_trigger)
    early_sorted = sorted(early, key=lambda s: abs(s.dist_atr))

    # Charts for signals
    for s in triggered_sorted:
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
    for s in early_sorted:
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
            f"<td style='padding-right:12px;'><img src='{vix_card}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'></td>"
            f"<td><img src='{eur_card}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'></td>"
            f"</tr></table>\n"
        )
    elif vix_card:
        md.append(f"<img src='{vix_card}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'>\n")
    elif eur_card:
        md.append(f"<img src='{eur_card}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'>\n")
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

    # Emerging chart trends (watchlist pulse) — before early callouts
    md.append(build_watchlist_pulse_section_md(
        df_early_new=df_early_new,
        df_early_old=df_early_old,
        df_trig_new=df_trig_new,
        df_trig_old=df_trig_old,
        watchlist_groups=WATCHLIST_GROUPS,
        ticker_labels=TICKER_LABELS,
    ))

    md.append("### 4A) Early callouts (~80% complete)\n")
    md.append("_Close enough to pre-plan. “Close enough” = within 0.5 ATR of the trigger (neckline/boundary). No SOFT tier — anything not CONFIRMED stays in EARLY._\n")
    md.append("**NEW (today):**\n")
    df_early_new_tbl = df_early_new.copy()
    if "Level" in df_early_new_tbl.columns and "Threshold" not in df_early_new_tbl.columns:
        df_early_new_tbl["Threshold"] = df_early_new_tbl["Level"]
    md.append(md_table_from_df(df_early_new_tbl, cols=["Ticker", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=40))
    # NEW early callouts: add a short, deterministic explanation + embed the annotated chart
    if df_early_new is not None and not df_early_new.empty:
        md.append("\n**What’s going on (NEW early callouts):**\n")
        # Keep it tight: show up to 8 explanations
        for _, rr in df_early_new.head(8).iterrows():
            t = str(rr.get("Ticker", "")).strip()
            sig_name = str(rr.get("Signal", "")).strip()
            try:
                close_v = float(rr.get("Close"))
            except Exception:
                close_v = float("nan")
            try:
                level_v = float(rr.get("Level"))
            except Exception:
                level_v = float("nan")
            try:
                dist_v = float(rr.get("Dist(ATR)"))
            except Exception:
                dist_v = float("nan")
            chart_p = rr.get("Chart", "")
            md.append(f"#### {display_ticker(t)} — `{sig_name}`")
            md.append(f"- **Trigger (level):** {level_v:.2f}  |  **Close:** {close_v:.2f}  |  **Distance:** {dist_v:+.2f} ATR")
            md.append("- Chart includes **SMA(50)** + **SMA(200)**, **volume**, plus trigger + confirmation (±0.5 ATR). HS/IHS is labeled (LS/H/RS) with neckline; Wedges include upper/lower trendlines with touch points.")
            # Pattern-specific blurb
            if "WEDGE" in sig_name:
                md.append("- **Wedge read:** upper/lower trendlines converge; chart marks recent touch points. Trigger is the boundary; confirmation is ±0.5 ATR beyond.")
            elif "HS_TOP" in sig_name or "IHS" in sig_name:
                md.append("- **HS/IHS read:** neckline is the trigger; chart labels LS/H/RS and draws the neckline + confirmation band.")
            else:
                md.append("- **Setup:** watch for confirmation close beyond trigger by ≥ 0.5 ATR, or a clean retest/failure in the direction of the signal.")

            if isinstance(chart_p, str) and chart_p:
                md.append(f'<img src="{chart_p}" width="720" style="max-width:100%;height:auto;">')
            md.append("")

    md.append("\n**ONGOING:**\n")
    df_early_old_tbl = df_early_old.copy()
    if "Level" in df_early_old_tbl.columns and "Threshold" not in df_early_old_tbl.columns:
        df_early_old_tbl["Threshold"] = df_early_old_tbl["Level"]
    md.append(md_table_from_df(df_early_old_tbl, cols=["Ticker", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=80))
    md.append("")

    md.append("### 4B) Breakouts / breakdowns (or about to)\n")
    md.append("_Includes **CONFIRMED** only: close beyond trigger by ≥0.5 ATR AND Volume ≥1.25×AvgVol(20) AND CLV ≥+0.70 (breakout) / ≤−0.70 (breakdown)._ \n")
    md.append("**NEW (today):**\n")
    df_trig_new_tbl = df_trig_new.copy()
    if "Level" in df_trig_new_tbl.columns and "Threshold" not in df_trig_new_tbl.columns:
        df_trig_new_tbl["Threshold"] = df_trig_new_tbl["Level"]
    md.append(md_table_from_df(df_trig_new_tbl, cols=["Ticker", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=60))
    md.append("\n**ONGOING:**\n")
    df_trig_old_tbl = df_trig_old.copy()
    if "Level" in df_trig_old_tbl.columns and "Threshold" not in df_trig_old_tbl.columns:
        df_trig_old_tbl["Threshold"] = df_trig_old_tbl["Level"]
    md.append(md_table_from_df(df_trig_old_tbl, cols=["Ticker", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=120))
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
    # Section 6: Full watchlist performance (grouped)
    md.append(build_watchlist_performance_section_md(WATCHLIST_GROUPS, ticker_labels=TICKER_LABELS, ticker_segment_rank=TICKER_SEGMENT_RANK))

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
