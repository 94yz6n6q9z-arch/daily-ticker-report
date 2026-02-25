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
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
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

# Friendly display-name overrides for report presentation
DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "000660.KS": "SK Hynix",
    "000660": "SK Hynix",
}

def _base_ticker(t: str) -> str:
    # Display ticker without exchange suffix (e.g., MC.PA -> MC, RRTL.DE -> RRTL)
    return t.split(".", 1)[0] if "." in t else t

def _display_name(t: str) -> str:
    t = str(t).strip()
    if t in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[t]
    base = _base_ticker(t)
    return DISPLAY_NAME_OVERRIDES.get(base, base)

# Ticker display labels: include segment tag when available, but hide exchange suffix.
TICKER_LABELS: Dict[str, str] = {t: f"{_display_name(t)} ({seg})" for t, seg in SEGMENT_TAGS.items()}

def display_ticker(t: str) -> str:
    # Use segment-tag label when available, otherwise strip exchange suffix (with overrides).
    return TICKER_LABELS.get(t, _display_name(t))


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
MSCI_WORLD_CLASSIFICATION_CSV = CONFIG_DIR / "msci_world_classification.csv"


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
# MSCI World classification (local CSV)
# ----------------------------
SP500_11_SECTORS: Tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)

_SECTOR_CANONICAL_MAP: Dict[str, str] = {
    "communication services": "Communication Services",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "energy": "Energy",
    "financials": "Financials",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "information technology": "Information Technology",
    "technology": "Information Technology",
    "it": "Information Technology",
    "materials": "Materials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
}

WATCHLIST_CATEGORY_BY_TICKER: Dict[str, str] = {}
for _cat_name, _tickers in WATCHLIST_GROUPS.items():
    for _t in _tickers:
        WATCHLIST_CATEGORY_BY_TICKER[str(_t).strip()] = _cat_name


def _normalize_sp500_sector_label(x: str) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    key = re.sub(r"\s+", " ", s).strip().lower()
    return _SECTOR_CANONICAL_MAP.get(key, s)


def load_msci_world_classification(path: Path = MSCI_WORLD_CLASSIFICATION_CSV) -> pd.DataFrame:
    """Load local MSCI World constituents + 11-sector classification CSV.

    Expected columns (flexible names): symbol/ticker, company/name (optional), country (optional), sector/category.
    Non-watchlist names should use one of the S&P 500 11 sector labels.
    """
    cols = ["Ticker", "Company", "Country", "Sector"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    try:
        raw = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"[msci] failed reading classification csv: {e}")
        return pd.DataFrame(columns=cols)

    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    def _pick(names: List[str]) -> Optional[str]:
        low = {str(c).strip().lower(): c for c in raw.columns}
        for n in names:
            if n in low:
                return low[n]
        return None

    col_t = _pick(["ticker", "symbol"])
    col_c = _pick(["company", "name", "security", "issuer"])
    col_s = _pick(["sector", "category", "gics_sector"])
    col_country = _pick(["country", "country_name", "country/region", "region_country"])
    if col_t is None or col_s is None:
        print("[msci] classification csv missing required columns: ticker/symbol and sector/category")
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    df["Ticker"] = df[col_t].astype(str).map(_clean_ticker).str.strip()
    df["Company"] = df[col_c].astype(str).str.strip() if col_c is not None else ""
    df["Country"] = df[col_country].astype(str).str.strip() if col_country is not None else ""
    df["Sector"] = df[col_s].astype(str).map(_normalize_sp500_sector_label).str.strip()
    df = df[(df["Ticker"] != "") & (df["Ticker"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["Ticker"], keep="first")

    invalid = sorted({s for s in df["Sector"].dropna().astype(str) if s and s not in SP500_11_SECTORS})
    if invalid:
        print(f"[msci] warning: {len(invalid)} sector labels not in S&P 500 11 sectors (examples: {invalid[:5]})")

    return df[cols].reset_index(drop=True)


def get_msci_world_tickers() -> List[str]:
    df = load_msci_world_classification(MSCI_WORLD_CLASSIFICATION_CSV)
    if df is None or df.empty:
        return []
    return sorted({str(x).strip() for x in df["Ticker"].astype(str).tolist() if str(x).strip()})


def build_category_resolver(msci_df: pd.DataFrame):
    msci_sector = {}
    if msci_df is not None and not msci_df.empty and "Ticker" in msci_df.columns:
        for _, r in msci_df.iterrows():
            t = str(r.get("Ticker", "")).strip()
            s = str(r.get("Sector", "")).strip()
            if t and s:
                msci_sector[t] = s
    def _resolve(ticker: str) -> str:
        t = str(ticker or "").strip()
        if not t:
            return ""
        if t in WATCHLIST_CATEGORY_BY_TICKER:
            return WATCHLIST_CATEGORY_BY_TICKER[t]
        base = _base_ticker(t)
        if base in WATCHLIST_CATEGORY_BY_TICKER:
            return WATCHLIST_CATEGORY_BY_TICKER[base]
        return msci_sector.get(t, msci_sector.get(base, "Unclassified"))

def _infer_country_from_ticker(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    # Exchange suffix heuristics (best-effort for watchlist names not in MSCI map)
    suffix = ""
    if "." in t:
        suffix = t.rsplit(".", 1)[-1]
    suffix_map = {
        "KS": "South Korea",
        "T": "Japan",
        "DE": "Germany",
        "MI": "Italy",
        "PA": "France",
        "SW": "Switzerland",
        "L": "United Kingdom",
        "MC": "Spain",
        "AS": "Netherlands",
        "HK": "Hong Kong",
        "TO": "Canada",
        "AX": "Australia",
        "ST": "Sweden",
        "CO": "Denmark",
        "HE": "Finland",
        "OL": "Norway",
        "BR": "Belgium",
    }
    if suffix in suffix_map:
        return suffix_map[suffix]
    # US/default for unsuffixed tickers in the watchlist-centric report
    return "United States"


def build_company_country_resolvers(msci_df: pd.DataFrame):
    msci_company: Dict[str, str] = {}
    msci_country: Dict[str, str] = {}
    if msci_df is not None and not msci_df.empty and "Ticker" in msci_df.columns:
        for _, r in msci_df.iterrows():
            t = str(r.get("Ticker", "")).strip()
            if not t:
                continue
            comp = str(r.get("Company", "") or "").strip()
            ctry = str(r.get("Country", "") or "").strip()
            if comp:
                msci_company[t] = comp
                msci_company.setdefault(_base_ticker(t), comp)
            if ctry:
                msci_country[t] = ctry
                msci_country.setdefault(_base_ticker(t), ctry)

    def _name(ticker: str) -> str:
        t = str(ticker or "").strip()
        if not t:
            return ""
        base = _base_ticker(t)
        return msci_company.get(t) or msci_company.get(base) or _display_name(t)

    def _country(ticker: str) -> str:
        t = str(ticker or "").strip()
        if not t:
            return ""
        base = _base_ticker(t)
        return msci_country.get(t) or msci_country.get(base) or _infer_country_from_ticker(t)

    return _name, _country


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


# ----------------------------
# Executive-summary headline selection (Yahoo + Investing + CNBC, market-only)
# ----------------------------
EXEC_SUMMARY_TARGET_SOURCES = ("Yahoo Finance", "Investing.com", "CNBC")

_EXEC_MARKET_KEYWORDS = [
    "market", "markets", "stock", "stocks", "share", "shares", "equity", "equities", "futures", "index", "indexes",
    "nasdaq", "s&p", "dow", "stoxx", "dax", "vix",
    "fed", "fomc", "powell", "rates", "rate cut", "yields", "yield", "treasury", "cpi", "pce", "inflation", "jobs", "payrolls",
    "earnings", "guidance", "results", "outlook",
    "ai", "nvidia", "semiconductor", "semiconductors", "chip", "chips",
    "dollar", "eur/usd", "euro", "fx", "currency",
    "oil", "crude", "brent", "wti", "energy",
    "bitcoin", "crypto", "ethereum",
    "tariff", "trade",
]

# Strong negatives to avoid random non-market headlines in top-news feeds.
_EXEC_NON_MARKET_KEYWORDS = [
    "sports", "soccer", "football", "nfl", "nba", "nhl", "mlb", "tennis", "golf",
    "celebrity", "movie", "movies", "tv", "music", "showbiz", "entertainment",
    "lifestyle", "fashion", "dating", "pregnant", "women", "royal", "crime", "murder", "trial",
]

_EXEC_THEME_STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "and", "on", "in", "as", "at", "by", "with", "from", "after",
    "before", "into", "over", "under", "amid", "ahead", "today", "live", "updates", "update", "why", "how",
    "what", "this", "that", "is", "are", "was", "were", "be", "it", "its", "their", "his", "her",
    "yahoo", "finance", "cnbc", "investing", "com",
}


def _rss_source_family(source_name: str) -> Optional[str]:
    s = (source_name or "").strip().lower()
    if "yahoo" in s:
        return "Yahoo Finance"
    if "invest" in s:
        return "Investing.com"
    if "cnbc" in s:
        return "CNBC"
    return None


def _market_headline_score(title: str, source_name: str = "", link: str = "") -> Tuple[int, List[str]]:
    t = (title or "").strip().lower()
    if not t:
        return (0, [])
    hits: List[str] = []
    score = 0

    for kw in _EXEC_MARKET_KEYWORDS:
        if kw in t:
            hits.append(kw)
            score += 3

    # Geopolitics only counts when clearly market-linked.
    geo = any(k in t for k in ["ukraine", "russia", "iran", "gaza", "middle east", "red sea"])
    geo_linked = any(k in t for k in ["oil", "crude", "gas", "energy", "shipping", "stocks", "markets", "futures"])
    if geo and geo_linked:
        hits.append("geo-linked")
        score += 4

    negative_hits = sum(1 for kw in _EXEC_NON_MARKET_KEYWORDS if kw in t)
    if negative_hits:
        score -= 6 * negative_hits

    src = (source_name or "").lower()
    if "cnbc markets" in src:
        score += 3
    if "yahoo finance" in src:
        score += 2
    if "investing.com" in src:
        score += 2

    # Guardrail: if geopolitics/news appears without clear market terms, penalize.
    if any(k in t for k in ["ukraine", "russia", "war", "gaza", "iran"]) and not any(
        k in t for k in ["market", "markets", "stocks", "futures", "oil", "crude", "yield", "rates"]
    ):
        score -= 8

    if score < 3:
        return (0, hits)
    return (score, hits)


def _headline_tokens(title: str) -> set:
    toks = re.findall(r"[A-Za-z0-9]+", (title or "").lower())
    out = set()
    for tok in toks:
        if len(tok) <= 2:
            continue
        if tok in _EXEC_THEME_STOPWORDS:
            continue
        out.add(tok)
    return out


def select_exec_summary_headlines(rss_items: List[Dict[str, str]]) -> Dict[str, object]:
    candidates: List[Dict[str, object]] = []
    for idx, it in enumerate(rss_items or []):
        title = (it.get("title", "") or "").strip()
        if not title:
            continue
        src_raw = (it.get("source", "") or "").strip()
        family = _rss_source_family(src_raw)
        if family not in EXEC_SUMMARY_TARGET_SOURCES:
            continue

        score, hits = _market_headline_score(title, src_raw, (it.get("link", "") or "").strip())
        if score <= 0:
            continue

        candidates.append({
            "source": src_raw,
            "source_family": family,
            "title": title,
            "link": (it.get("link", "") or "").strip(),
            "pubDate": (it.get("pubDate", "") or "").strip(),
            "market_score": int(score),
            "keyword_hits": hits[:8],
            "_idx": idx,
        })

    by_family: Dict[str, List[Dict[str, object]]] = {k: [] for k in EXEC_SUMMARY_TARGET_SOURCES}
    for c in candidates:
        by_family[str(c["source_family"])].append(c)

    for fam in by_family:
        by_family[fam].sort(key=lambda x: (-int(x.get("market_score", 0)), int(x.get("_idx", 99999))))

    selected: List[Dict[str, object]] = []
    selected_by_source: List[Dict[str, str]] = []
    for fam in EXEC_SUMMARY_TARGET_SOURCES:
        if by_family[fam]:
            c = by_family[fam][0]
            selected.append(c)
            selected_by_source.append({
                "source_family": fam,
                "source": str(c.get("source", "")),
                "title": str(c.get("title", "")),
                "link": str(c.get("link", "")),
                "market_score": str(c.get("market_score", "")),
            })

    extras: List[Dict[str, object]] = []
    for fam in EXEC_SUMMARY_TARGET_SOURCES:
        extras.extend(by_family[fam][1:3])
    extras.sort(key=lambda x: (-int(x.get("market_score", 0)), int(x.get("_idx", 99999))))
    for c in extras:
        if len(selected) >= 6:
            break
        selected.append(c)

    if not selected:
        for idx, it in enumerate(rss_items or []):
            src_raw = (it.get("source", "") or "").strip()
            fam = _rss_source_family(src_raw)
            title = (it.get("title", "") or "").strip()
            if fam in EXEC_SUMMARY_TARGET_SOURCES and title:
                selected.append({
                    "source": src_raw,
                    "source_family": fam,
                    "title": title,
                    "link": (it.get("link", "") or "").strip(),
                    "pubDate": (it.get("pubDate", "") or "").strip(),
                    "market_score": 1,
                    "keyword_hits": [],
                    "_idx": idx,
                })
                selected_by_source.append({
                    "source_family": fam,
                    "source": src_raw,
                    "title": title,
                    "link": (it.get("link", "") or "").strip(),
                    "market_score": "1",
                })
                break

    dominant: Optional[Dict[str, object]] = None
    if selected:
        toks = [_headline_tokens(str(c.get("title", ""))) for c in selected]
        ranks: List[Tuple[int, int]] = []
        for i, c in enumerate(selected):
            base_score = int(c.get("market_score", 0))
            overlap_sources = 0
            overlap_tokens = 0
            fam_i = str(c.get("source_family", ""))
            for j, c2 in enumerate(selected):
                if i == j:
                    continue
                fam_j = str(c2.get("source_family", ""))
                inter = toks[i].intersection(toks[j])
                if inter:
                    overlap_tokens += len(inter)
                    if fam_i != fam_j:
                        overlap_sources += 1
            total = base_score + (4 * overlap_sources) + min(overlap_tokens, 6)
            ranks.append((total, i))
        ranks.sort(reverse=True)
        dominant = selected[ranks[0][1]]

    selected_out = [{
        "source": str(c.get("source", "")),
        "source_family": str(c.get("source_family", "")),
        "title": str(c.get("title", "")),
        "link": str(c.get("link", "")),
        "pubDate": str(c.get("pubDate", "")),
        "market_score": str(c.get("market_score", "")),
    } for c in selected]

    dominant_out = None
    if dominant:
        dominant_out = {
            "source": str(dominant.get("source", "")),
            "source_family": str(dominant.get("source_family", "")),
            "title": str(dominant.get("title", "")),
            "link": str(dominant.get("link", "")),
            "pubDate": str(dominant.get("pubDate", "")),
            "market_score": str(dominant.get("market_score", "")),
        }

    return {
        "selected_headlines": selected_out,
        "selected_by_source": selected_by_source,
        "dominant_headline": dominant_out,
        "coverage": {
            "Yahoo Finance": len(by_family.get("Yahoo Finance", [])),
            "Investing.com": len(by_family.get("Investing.com", [])),
            "CNBC": len(by_family.get("CNBC", [])),
        },
    }



def _normalize_openai_model_for_api(model: str) -> str:
    """Normalize ChatGPT-style model labels to API model IDs.

    In ChatGPT, users may think in terms of "GPT-5.2 Thinking" / "Instant".
    In the API, the main reasoning model is `gpt-5.2` (with `reasoning.effort`).
    """
    m = (model or "").strip()
    if not m:
        return ""
    ml = m.lower()
    aliases = {
        "gpt-5.2-thinking": "gpt-5.2",
        "gpt-5.2-think": "gpt-5.2",
        "gpt-5.2-instant": "gpt-5.2",
        "gpt-5.2-default": "gpt-5.2",
        "gpt5.2-thinking": "gpt-5.2",
        "gpt5.2": "gpt-5.2",
    }
    return aliases.get(ml, m)


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

    preferred_raw = (os.environ.get("OPENAI_MODEL", "") or "").strip()
    preferred = _normalize_openai_model_for_api(preferred_raw)
    if preferred_raw and preferred_raw != preferred:
        print(f"[openai] normalized OPENAI_MODEL {preferred_raw} -> {preferred}")

    # Note: ChatGPT labels (e.g., "GPT-5.2 Thinking") do not always match API model IDs.
    # In the API, `gpt-5.2` + reasoning.effort is the main reasoning path.
    candidates = [m for m in [preferred, "gpt-5.2", "gpt-5.2-pro", "gpt-4.1", "gpt-4o"] if m]
    seen = set(); models = []
    for m in candidates:
        m2 = _normalize_openai_model_for_api(m)
        if m2 not in seen:
            models.append(m2); seen.add(m2)

    effort = (os.environ.get("OPENAI_REASONING_EFFORT", "medium") or "medium").strip()

    instructions = """You are an experienced Financial Times markets editor.

Task: Write the Executive summary for a daily market report.

Output EXACTLY 2 or 3 sentences (no bullets, no headings).
Format rules:
- Sentence 1 must start with the provided THEME_PHRASE followed by a colon (normally "Headline:").
- Sentence 1 should be a SYNTHESIZED market-theme headline in your own words (not a copied article title).
- Sentence 2 should cover key market performance and context.
- Sentence 3 (or the end of sentence 2 if only 2 sentences) should mention biggest movers >4% on either side.

Hard rules:
A) Use ONLY the provided market data + the provided selected headlines; do not invent events, names, or catalysts.
B) Build the headline theme from the cross-source market headlines selected from Yahoo Finance, Investing.com, and CNBC.
   Ignore non-market/general-interest headlines even if they exist elsewhere in the payload.
C) The market-performance sentence must include at least NDX 1D, S&P 1D, and VIX 1D.
D) Contextualize today inside the last 3–4 weeks as a narrative (continuation/reversal of the recent tape).
   - You MAY use 7D/1M stats only as brief supporting evidence (max ONE short parenthetical).
   - Do NOT write a horizon-comparison sentence like “Over the past month vs three months …”.
E) Mention watchlist movers ≥4% on BOTH sides if present (up to 2 gainers + 2 losers). If none, say so.
F) Use provided mover labels verbatim (e.g., "SK Hynix", not raw ticker codes) when available.
G) Only mention oil/FX when justified by headlines or clear linkage; otherwise omit.

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
        except HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            err_body = re.sub(r"\s+", " ", err_body).strip()[:1200]
            if err_body:
                print(f"[openai] exec model={model} minimal={minimal} failed: HTTP Error {e.code}: {e.reason} | body={err_body}")
            else:
                print(f"[openai] exec model={model} minimal={minimal} failed: HTTP Error {e.code}: {e.reason}")
            return None
        except URLError as e:
            print(f"[openai] exec model={model} minimal={minimal} failed: URLError {e}")
            return None
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
    md.append("### 4) Watchlist emerging chart trends")
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
    # Table-only by user preference (no narrative bullets below the table).
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

    # Executive-summary headline context: select only market-relevant headlines from Yahoo + Investing + CNBC.
    headline_ctx = select_exec_summary_headlines(rss_items or [])
    top_headlines = headline_ctx.get("selected_headlines", []) if isinstance(headline_ctx, dict) else []
    dominant = headline_ctx.get("dominant_headline") if isinstance(headline_ctx, dict) else None
    # Keep the user-facing structure stable: "Headline: ..." (the prose after the colon comes from OpenAI).
    theme_phrase = "Headline"

    def _fmt_exec_movers(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for t, p in (items or []):
            try:
                out.append((display_ticker(str(t)), float(p)))
            except Exception:
                try:
                    out.append((display_ticker(str(t)), p))
                except Exception:
                    out.append((str(t), p))
        return out

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
            "session": _fmt_exec_movers(watchlist_movers.get("session", [])),
            "after_hours": _fmt_exec_movers(watchlist_movers.get("after_hours", [])),
        },
        "headline_themes": summarize_rss_themes(top_headlines if top_headlines else rss_items),
        "dominant_headline": dominant,
        "selected_top_market_headlines_by_source": (headline_ctx or {}).get("selected_by_source", []),
        "headline_selection_debug": {
            "coverage": (headline_ctx or {}).get("coverage", {}),
            "selected_headlines": top_headlines,
        },
        "theme_phrase": theme_phrase,
        "headlines": top_headlines,
    }

    _exec_debug_on = str(os.environ.get("EXEC_SUMMARY_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    if _exec_debug_on:
        try:
            print("[exec_summary][headline_debug] coverage=", json.dumps((headline_ctx or {}).get("coverage", {}), ensure_ascii=False))
            print("[exec_summary][headline_debug] dominant=", json.dumps(dominant, ensure_ascii=False))
            print("[exec_summary][headline_debug] selected_by_source=", json.dumps((headline_ctx or {}).get("selected_by_source", []), ensure_ascii=False))
        except Exception:
            pass
    else:
        print("[exec_summary][headline_debug] disabled (set EXEC_SUMMARY_DEBUG=1 to log selected headlines)")

    llm = _openai_responses_exec_summary(json.dumps(payload, ensure_ascii=False))
    if llm:
        # Enforce the user-requested opener so the first sentence always starts with "Headline:".
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
    # Keep the same structure: Headline / key market performance / movers >4%.
    dom_title = str((dominant or {}).get("title", "")).strip()
    dom_source = str((dominant or {}).get("source_family") or (dominant or {}).get("source") or "").strip()

    if dom_title:
        headline_text = dom_title
        # Clean and shorten raw headline into a more report-friendly line.
        headline_text = re.sub(r"\s+", " ", headline_text).strip().rstrip(".?!")
        if len(headline_text) > 110:
            headline_text = headline_text[:107].rstrip() + "..."
        if dom_source:
            s1 = f"{theme_phrase}: {headline_text} ({dom_source})."
        else:
            s1 = f"{theme_phrase}: {headline_text}."
    else:
        s1 = f"{theme_phrase}: Markets were driven by a mix of macro, rates and company-specific headlines across Yahoo Finance, Investing.com and CNBC."

    movers = watchlist_movers.get("session", []) + watchlist_movers.get("after_hours", [])
    gainers = sorted([x for x in movers if x[1] >= MOVER_THRESHOLD_PCT], key=lambda z: z[1], reverse=True)
    losers = sorted([x for x in movers if x[1] <= -MOVER_THRESHOLD_PCT], key=lambda z: z[1])

    s2 = (
        f"The Nasdaq rose {f(ndx,'1D'):+.1f}% and the S&P 500 {'rose' if f(spx,'1D') >= 0 else 'fell'} {abs(f(spx,'1D')):.1f}%, while the VIX {'fell' if f(vix,'1D') <= 0 else 'rose'} {abs(f(vix,'1D')):.1f}%, with the session extending a choppy recent tape"
        + (f" (NDX {f(ndx,'1M'):+.1f}% over 1M)." if not math.isnan(f(ndx,'1M')) else ".")
    )

    if not gainers and not losers:
        s3 = "No watchlist names moved more than 4% (including after-hours)."
    else:
        def _fmt(items: List[Tuple[str, float]]) -> str:
            return ", ".join([f"{display_ticker(str(t))} ({float(p):+,.1f}%)" for t, p in items])
        parts = []
        if gainers:
            parts.append(_fmt(gainers[:2]))
        if losers:
            parts.append(_fmt(losers[:2]))
        s3 = "Watchlist movers >4% included " + ("; ".join(parts)) + "."

    return s1 + " " + s2 + " " + s3


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
# Technical patterns (deterministic rules engine)
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
    meta: Optional[Dict[str, Any]] = None


@dataclass
class PatternCandidate:
    pattern: str
    direction: str   # BREAKOUT / BREAKDOWN
    level: float
    meta: Optional[Dict[str, Any]] = None


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _median_close(df: pd.DataFrame, start: int = 0, end: Optional[int] = None) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan")
    s = pd.to_numeric(df["Close"].iloc[start:end], errors="coerce").dropna()
    return float(s.median()) if not s.empty else float("nan")


def _median_atr(df: pd.DataFrame, start: int = 0, end: Optional[int] = None) -> float:
    try:
        a = atr(df, ATR_N)
        s = pd.to_numeric(a.iloc[start:end], errors="coerce").dropna()
        if not s.empty:
            return float(s.median())
    except Exception:
        pass
    # Fallback if ATR unavailable
    mc = _median_close(df, start, end)
    if pd.notna(mc):
        return max(mc * 0.01, 1e-6)
    return 1e-6


def _pivot_tolerance(df: pd.DataFrame, start: int = 0, end: Optional[int] = None) -> float:
    atr_med = _median_atr(df, start, end)
    close_med = _median_close(df, start, end)
    cterm = 0.0075 * close_med if pd.notna(close_med) else 0.0
    return float(max(0.35 * atr_med, cterm, 1e-6))


def _swing_points(series: pd.Series, window: int = 3) -> Tuple[List[int], List[int]]:
    """Legacy close-based pivots (kept for backwards compatibility / fallbacks)."""
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


def _swing_points_ohlc(
    df: pd.DataFrame,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    Deterministic pivots on High/Low with prominence filter:
    - swing high: local max in [i-window, i+window], unique, prominence >= 0.5 ATR
    - swing low: local min analogously
    """
    dd = df.dropna(subset=["High", "Low", "Close"]).copy()
    if dd.empty or len(dd) < (2 * window + 5):
        return [], []
    hi = dd["High"].astype(float).values
    lo = dd["Low"].astype(float).values
    atr_s = atr(dd, ATR_N)
    atr_v = pd.to_numeric(atr_s, errors="coerce").values if atr_s is not None else np.full(len(dd), np.nan)

    highs: List[int] = []
    lows: List[int] = []

    for i in range(window, len(dd) - window):
        hwin = hi[i - window : i + window + 1]
        lwin = lo[i - window : i + window + 1]
        if np.isnan(hwin).any() or np.isnan(lwin).any():
            continue

        atr_i = atr_v[i] if i < len(atr_v) and np.isfinite(atr_v[i]) else np.nan
        if not np.isfinite(atr_i):
            lo_i = max(0, i - 20)
            atr_i = np.nanmedian(atr_v[lo_i : i + 1]) if np.isfinite(np.nanmedian(atr_v[lo_i : i + 1])) else np.nan
        if not np.isfinite(atr_i):
            atr_i = max(float(np.nanmedian((hwin - lwin))), 1e-6)

        # High pivot
        if hi[i] == np.max(hwin) and np.sum(hwin == hi[i]) == 1:
            prominence = float(hi[i] - np.min(lwin))
            if prominence >= float(prominence_atr_mult * atr_i):
                highs.append(i)

        # Low pivot
        if lo[i] == np.min(lwin) and np.sum(lwin == lo[i]) == 1:
            prominence = float(np.max(hwin) - lo[i])
            if prominence >= float(prominence_atr_mult * atr_i):
                lows.append(i)

    return highs, lows


def _line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return (0.0, float(y[-1]) if len(y) else 0.0)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _line_eval(a: float, b: float, x: float) -> float:
    return float(a * x + b)


def _trend_context_label(c: pd.Series, pattern_start: int, atr_med: float) -> str:
    """
    Prior-trend classifier for top/bottom labeling:
      - TOP if pre-window slope >0 and net move > +2 ATR
      - BOTTOM if slope <0 and net move < -2 ATR
      - else NEUTRAL
    """
    if c is None or c.empty:
        return "NEUTRAL"
    end = max(0, pattern_start)
    start = max(0, end - 40)
    if end - start < 20:
        return "NEUTRAL"
    seg = pd.to_numeric(c.iloc[start:end], errors="coerce").dropna()
    if len(seg) < 20:
        return "NEUTRAL"
    xs = np.arange(len(seg), dtype=float)
    try:
        a, _ = np.polyfit(xs, seg.values.astype(float), 1)
    except Exception:
        a = 0.0
    net = float(seg.iloc[-1] - seg.iloc[0])
    thresh = max(2.0 * float(atr_med), 1e-6)
    if a > 0 and net > thresh:
        return "TOP"
    if a < 0 and net < -thresh:
        return "BOTTOM"
    return "NEUTRAL"


def _horizontal_slope_threshold(df: pd.DataFrame, start: int = 0, end: Optional[int] = None) -> float:
    # "horizontal" if |slope| <= 0.05*ATR per bar
    atr_med = _median_atr(df, start, end)
    return float(max(0.05 * atr_med, 1e-8))


def _touch_indices_for_line(
    pivot_indices: List[int],
    pivot_prices: np.ndarray,
    a: float,
    b: float,
    tol: float,
) -> List[int]:
    out: List[int] = []
    for idx, px in zip(pivot_indices, pivot_prices):
        if abs(float(px) - _line_eval(a, b, float(idx))) <= tol:
            out.append(int(idx))
    return out


def _alternation_count(events: List[Tuple[int, str]]) -> int:
    if not events:
        return 0
    events = sorted(events, key=lambda x: x[0])
    cnt = 0
    prev = None
    for _, side in events:
        if prev is None:
            prev = side
            continue
        if side != prev:
            cnt += 1
            prev = side
    return cnt


def _iso_ts(idx_val) -> str:
    try:
        return pd.Timestamp(idx_val).isoformat()
    except Exception:
        return str(idx_val)


def _point_meta(df: pd.DataFrame, i: int, price: float, label: str, kind: str = "point") -> Dict[str, Any]:
    return {"t": _iso_ts(df.index[i]), "p": float(price), "label": str(label), "kind": kind, "i": int(i)}


def _line_meta(df: pd.DataFrame, i1: int, y1: float, i2: int, y2: float, label: str) -> Dict[str, Any]:
    return {
        "t1": _iso_ts(df.index[i1]), "y1": float(y1),
        "t2": _iso_ts(df.index[i2]), "y2": float(y2),
        "label": str(label), "i1": int(i1), "i2": int(i2)
    }


def _build_band_pattern_meta(
    df: pd.DataFrame,
    pattern: str,
    start_i: int,
    end_i: int,
    a_u: float,
    b_u: float,
    a_l: float,
    b_l: float,
    hi_touches: List[int],
    lo_touches: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "annot_type": "band",
        "pattern": pattern,
        "start_i": int(start_i),
        "end_i": int(end_i),
        "lines": [
            _line_meta(df, start_i, _line_eval(a_u, b_u, start_i), end_i, _line_eval(a_u, b_u, end_i), "Upper"),
            _line_meta(df, start_i, _line_eval(a_l, b_l, start_i), end_i, _line_eval(a_l, b_l, end_i), "Lower"),
        ],
        "touch_points": [],
    }
    for i in hi_touches:
        meta["touch_points"].append(_point_meta(df, i, float(df["High"].iloc[i]), "H touch", kind="touch_high"))
    for i in lo_touches:
        meta["touch_points"].append(_point_meta(df, i, float(df["Low"].iloc[i]), "L touch", kind="touch_low"))
    if extra:
        meta.update(extra)
    return meta


def _pick_recent_hs_triplet(
    highs_idx: List[int],
    lows_idx: List[int],
    c: pd.Series,
    d: pd.DataFrame,
    inverse: bool = False,
) -> Optional[Tuple[int, int, int, int, int, float, float, float]]:
    """
    Returns (p1, p2, p3, t1, t2, px1, px2, px3) for H&S/IHS candidate if rules pass.
    """
    if len(highs_idx) + len(lows_idx) < 5:
        return None
    piv_hi = highs_idx[-16:]
    piv_lo = lows_idx[-16:]
    if inverse:
        pivots_primary = piv_lo
        pivots_between = piv_hi
    else:
        pivots_primary = piv_hi
        pivots_between = piv_lo

    if len(pivots_primary) < 3 or len(pivots_between) < 2:
        return None

    best = None
    best_score = -1e18

    for a_i in range(0, len(pivots_primary) - 2):
        for b_i in range(a_i + 1, len(pivots_primary) - 1):
            for c_i in range(b_i + 1, len(pivots_primary)):
                p1, p2, p3 = pivots_primary[a_i], pivots_primary[b_i], pivots_primary[c_i]
                # Need intervening opposite pivots
                between1 = [x for x in pivots_between if p1 < x < p2]
                between2 = [x for x in pivots_between if p2 < x < p3]
                if not between1 or not between2:
                    continue

                px1 = float(c.iloc[p1]); px2 = float(c.iloc[p2]); px3 = float(c.iloc[p3])

                # Time symmetry
                dL = max(1, p2 - p1); dR = max(1, p3 - p2)
                ratio = dL / dR
                if ratio < 0.5 or ratio > 2.0:
                    continue

                # Pattern ATR context
                start_i = p1
                end_i = p3 + 1
                atr_med = _median_atr(d, start_i, end_i)
                price_ref = float(np.nanmedian([px1, px2, px3]))
                min_head_gap = max(0.5 * atr_med, 0.02 * max(price_ref, 1e-6))
                shoulder_tol = max(1.0 * atr_med, 0.05 * max((px1 + px3) / 2.0, 1e-6))

                # Head / shoulders geometry
                if inverse:
                    if not (px2 <= min(px1, px3) - min_head_gap):
                        continue
                    if abs(px1 - px3) > shoulder_tol:
                        continue
                    # Highest highs between shoulders/head define neckline points
                    t1 = max(between1, key=lambda k: float(d["High"].iloc[k]))
                    t2 = max(between2, key=lambda k: float(d["High"].iloc[k]))
                    n1 = float(d["High"].iloc[t1]); n2 = float(d["High"].iloc[t2])
                    head_gap_quality = min(px1 - px2, px3 - px2)
                else:
                    if not (px2 >= max(px1, px3) + min_head_gap):
                        continue
                    if abs(px1 - px3) > shoulder_tol:
                        continue
                    t1 = min(between1, key=lambda k: float(d["Low"].iloc[k]))
                    t2 = min(between2, key=lambda k: float(d["Low"].iloc[k]))
                    n1 = float(d["Low"].iloc[t1]); n2 = float(d["Low"].iloc[t2])
                    head_gap_quality = min(px2 - px1, px2 - px3)

                # Prior trend label enforcement
                trend = _trend_context_label(c, p1, atr_med)
                if inverse and trend != "BOTTOM":
                    continue
                if (not inverse) and trend != "TOP":
                    continue

                # Score: recency + symmetry + prominence
                recency_bonus = float(p3)
                sym_penalty = abs(math.log(ratio))
                neck_span = abs(n2 - n1)
                score = recency_bonus + 4.0 * (head_gap_quality / max(atr_med, 1e-6)) - 2.0 * sym_penalty - 0.25 * (neck_span / max(atr_med, 1e-6))
                if score > best_score:
                    best_score = score
                    best = (p1, p2, p3, int(t1), int(t2), px1, px2, px3)

    return best


def detect_hs_top(df: pd.DataFrame) -> Optional[PatternCandidate]:
    d = df.tail(LOOKBACK_DAYS).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 120:
        return None
    c = d["Close"].astype(float)
    highs_idx, lows_idx = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5)
    if len(highs_idx) < 3 or len(lows_idx) < 2:
        return None

    hs = _pick_recent_hs_triplet(highs_idx, lows_idx, c, d, inverse=False)
    if hs is None:
        return None
    p1, p2, p3, t1, t2, px1, px2, px3 = hs

    # Neckline through troughs (sloped allowed)
    n1 = float(d["Low"].iloc[t1]); n2 = float(d["Low"].iloc[t2])
    a_n, b_n = _line_fit(np.array([float(t1), float(t2)]), np.array([n1, n2]))
    x_last = float(len(d) - 1)
    neckline_now = _line_eval(a_n, b_n, x_last)

    meta = {
        "annot_type": "hs",
        "variant": "top",
        "points": [
            _point_meta(d, p1, px1, "LS"),
            _point_meta(d, p2, px2, "H"),
            _point_meta(d, p3, px3, "RS"),
            _point_meta(d, t1, n1, "T1"),
            _point_meta(d, t2, n2, "T2"),
        ],
        "lines": [
            _line_meta(d, t1, n1, t2, n2, "Neckline"),
        ],
        "pattern_start_i": int(p1),
        "pattern_end_i": int(p3),
        "trigger_line_type": "neckline",
    }
    return PatternCandidate(pattern="HS_TOP", direction="BREAKDOWN", level=float(neckline_now), meta=meta)


def detect_inverse_hs(df: pd.DataFrame) -> Optional[PatternCandidate]:
    d = df.tail(LOOKBACK_DAYS).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 120:
        return None
    c = d["Close"].astype(float)
    highs_idx, lows_idx = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5)
    if len(lows_idx) < 3 or len(highs_idx) < 2:
        return None

    ihs = _pick_recent_hs_triplet(highs_idx, lows_idx, c, d, inverse=True)
    if ihs is None:
        return None
    p1, p2, p3, r1, r2, px1, px2, px3 = ihs

    h1 = float(d["High"].iloc[r1]); h2 = float(d["High"].iloc[r2])
    a_n, b_n = _line_fit(np.array([float(r1), float(r2)]), np.array([h1, h2]))
    atr_med = _median_atr(d, p1, p3 + 1)
    # "Too steep neckline" safeguard -> horizontal trigger at higher reaction high
    if abs(a_n) > 0.20 * max(atr_med, 1e-6):
        use_horiz = True
        neckline_now = float(max(h1, h2))
        line_for_meta = _line_meta(d, min(r1, r2), neckline_now, len(d) - 1, neckline_now, "Neckline")
    else:
        use_horiz = False
        x_last = float(len(d) - 1)
        neckline_now = _line_eval(a_n, b_n, x_last)
        line_for_meta = _line_meta(d, r1, h1, r2, h2, "Neckline")

    meta = {
        "annot_type": "hs",
        "variant": "inverse",
        "points": [
            _point_meta(d, p1, px1, "LS"),
            _point_meta(d, p2, px2, "H"),
            _point_meta(d, p3, px3, "RS"),
            _point_meta(d, r1, h1, "R1"),
            _point_meta(d, r2, h2, "R2"),
        ],
        "lines": [line_for_meta],
        "pattern_start_i": int(p1),
        "pattern_end_i": int(p3),
        "trigger_line_type": "neckline_horizontal" if use_horiz else "neckline",
    }
    return PatternCandidate(pattern="IHS", direction="BREAKOUT", level=float(neckline_now), meta=meta)


def _detect_band_structure(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Deterministic detector for rectangles / broadening / triangles.

    Returns a dict describing a geometric band (upper/lower lines + metadata),
    or None if no valid structure is present.
    """
    d = df.tail(180).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 100:
        return None
    c = d["Close"].astype(float)
    highs_idx, lows_idx = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5)
    if len(highs_idx) < 4 or len(lows_idx) < 4:
        return None

    hi_piv = highs_idx[-8:]
    lo_piv = lows_idx[-8:]
    if len(hi_piv) < 4 or len(lo_piv) < 4:
        return None

    # Fit on the most recent 4–6 pivots per side for stability.
    hi_fit = hi_piv[-6:] if len(hi_piv) >= 6 else hi_piv[-4:]
    lo_fit = lo_piv[-6:] if len(lo_piv) >= 6 else lo_piv[-4:]

    xh = np.array(hi_fit, dtype=float)
    yh = np.array([float(d["High"].iloc[i]) for i in hi_fit], dtype=float)
    xl = np.array(lo_fit, dtype=float)
    yl = np.array([float(d["Low"].iloc[i]) for i in lo_fit], dtype=float)

    a_u, b_u = _line_fit(xh, yh)
    a_l, b_l = _line_fit(xl, yl)

    start_i = int(max(0, min(int(min(hi_fit)), int(min(lo_fit))) - 2))
    end_i = int(len(d) - 1)
    if end_i - start_i < 20:
        return None

    width_start = _line_eval(a_u, b_u, float(start_i)) - _line_eval(a_l, b_l, float(start_i))
    width_end = _line_eval(a_u, b_u, float(end_i)) - _line_eval(a_l, b_l, float(end_i))
    if not (np.isfinite(width_start) and np.isfinite(width_end)):
        return None
    if width_start <= 0 or width_end <= 0:
        return None

    atr_med = _median_atr(d, start_i, end_i + 1)
    tol = _pivot_tolerance(d, start_i, end_i + 1)
    slope_horiz = _horizontal_slope_threshold(d, start_i, end_i + 1)
    close_med = _median_close(d, start_i, end_i + 1)

    # Touch counts from all pivots in pattern window
    hi_all = [i for i in highs_idx if start_i <= i <= end_i]
    lo_all = [i for i in lows_idx if start_i <= i <= end_i]
    hi_all_prices = np.array([float(d["High"].iloc[i]) for i in hi_all], dtype=float) if hi_all else np.array([])
    lo_all_prices = np.array([float(d["Low"].iloc[i]) for i in lo_all], dtype=float) if lo_all else np.array([])
    hi_touches = _touch_indices_for_line(hi_all, hi_all_prices, a_u, b_u, tol) if len(hi_all_prices) else []
    lo_touches = _touch_indices_for_line(lo_all, lo_all_prices, a_l, b_l, tol) if len(lo_all_prices) else []

    if len(hi_touches) < 2 or len(lo_touches) < 2:
        return None

    # Traversals / alternation (triangles need multiple side-to-side moves)
    touch_events = [(i, "U") for i in hi_touches] + [(i, "L") for i in lo_touches]
    alternations = _alternation_count(touch_events)

    # Containment (for rectangles quality)
    seg_close = c.iloc[start_i : end_i + 1]
    inside = 0
    total = 0
    for j, px in enumerate(seg_close.values, start=start_i):
        up = _line_eval(a_u, b_u, float(j)) + tol
        lo_ = _line_eval(a_l, b_l, float(j)) - tol
        if np.isfinite(px) and np.isfinite(up) and np.isfinite(lo_):
            total += 1
            if lo_ <= float(px) <= up:
                inside += 1
    containment = (inside / total) if total > 0 else 0.0

    # Trend label for top/bottom variants
    trend_label = _trend_context_label(c, start_i, atr_med)

    # Converging/diverging
    converging = width_end <= 0.80 * width_start
    diverging = width_end >= 1.20 * width_start

    # Apex for converging structures
    apex_x = None
    if abs(a_u - a_l) > 1e-10:
        apex_x = (b_l - b_u) / (a_u - a_l)

    # Triangle progress to apex (0 at start, 1 at apex)
    progress = None
    if apex_x is not None and np.isfinite(apex_x) and apex_x > start_i:
        progress = (end_i - start_i) / max(apex_x - start_i, 1e-9)

    # Pattern classification
    pat = None
    extra: Dict[str, Any] = {
        "containment": float(containment),
        "alternations": int(alternations),
        "trend_label": trend_label,
    }

    upper_horizontal = abs(a_u) <= slope_horiz
    lower_horizontal = abs(a_l) <= slope_horiz

    # Rectangle (top/bottom by prior trend)
    rect_height = min(width_start, width_end)
    if (
        upper_horizontal and lower_horizontal
        and containment >= 0.80
        and rect_height >= 1.0 * max(atr_med, 1e-6)
    ):
        if trend_label == "TOP":
            pat = "RECT_TOP"
        elif trend_label == "BOTTOM":
            pat = "RECT_BOTTOM"
        else:
            pat = "RECT"
    # Broadening (megaphone): higher highs + lower lows / diverging lines
    elif (
        diverging
        and a_u > slope_horiz
        and a_l < -slope_horiz
        and len(hi_touches) >= 2 and len(lo_touches) >= 2
        and (width_end >= 1.2 * width_start)
    ):
        if trend_label == "TOP":
            pat = "BROADEN_TOP"
        elif trend_label == "BOTTOM":
            pat = "BROADEN_BOTTOM"
        else:
            pat = "BROADEN"
    # Triangles (ascending / descending / symmetrical)
    elif converging and (apex_x is not None) and np.isfinite(apex_x):
        apex_in_future = apex_x > end_i and apex_x <= (end_i + 3.0 * max(end_i - start_i, 1))
        if apex_in_future and len(hi_touches) >= 2 and len(lo_touches) >= 2 and alternations >= 3:
            extra["apex_x"] = float(apex_x)
            extra["progress_to_apex"] = float(progress) if progress is not None else np.nan
            # Prefer triangles in the "watchable" part of the pattern, but do not over-restrict confirmed breaks.
            if progress is not None and progress < 0.35:
                return None

            if upper_horizontal and a_l > slope_horiz:
                pat = "ASC_TRIANGLE"
            elif lower_horizontal and a_u < -slope_horiz:
                pat = "DESC_TRIANGLE"
            elif a_u < -slope_horiz and a_l > slope_horiz:
                if trend_label == "TOP":
                    pat = "SYM_TRIANGLE_TOP"
                elif trend_label == "BOTTOM":
                    pat = "SYM_TRIANGLE_BOTTOM"
                else:
                    pat = "SYM_TRIANGLE"
            else:
                pat = None

    if pat is None:
        return None

    meta = _build_band_pattern_meta(
        d, pat, start_i, end_i, a_u, b_u, a_l, b_l, hi_touches, lo_touches, extra=extra
    )
    return {
        "pattern": pat,
        "df": d,
        "a_u": float(a_u), "b_u": float(b_u),
        "a_l": float(a_l), "b_l": float(b_l),
        "start_i": int(start_i), "end_i": int(end_i),
        "upper_level": float(_line_eval(a_u, b_u, float(end_i))),
        "lower_level": float(_line_eval(a_l, b_l, float(end_i))),
        "meta": meta,
    }


def detect_structure_candidates(df: pd.DataFrame) -> List[PatternCandidate]:
    out: List[PatternCandidate] = []
    st = _detect_band_structure(df)
    if not st:
        return out
    pat = str(st["pattern"])
    upper_level = float(st["upper_level"])
    lower_level = float(st["lower_level"])
    base_meta = st.get("meta") or {}
    # Two-sided triggers for band structures (even for asc/desc triangles, failures can trade)
    out.append(PatternCandidate(pattern=pat, direction="BREAKOUT", level=upper_level, meta=dict(base_meta)))
    out.append(PatternCandidate(pattern=pat, direction="BREAKDOWN", level=lower_level, meta=dict(base_meta)))
    return out


def detect_dead_cat_bounce(df: pd.DataFrame) -> Optional[PatternCandidate]:
    """
    Deterministic DCB detector:
      - gap-down event (strict gap OR >3% gap-down open)
      - plunge >=20% from pre-event high to event low within 1-3 days
      - event volume >=1.5x avg20
      - bounce retrace 10%-60%
      - rollover before current bar
      - trigger = aggressive (post-bounce swing low) else conservative (event low)
    """
    d = df.tail(140).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 50:
        return None
    if "Volume" not in d.columns or d["Volume"].dropna().empty:
        return None

    H = d["High"].astype(float).values
    L = d["Low"].astype(float).values
    O = d["Open"].astype(float).values
    C = d["Close"].astype(float).values
    V = pd.to_numeric(d["Volume"], errors="coerce").astype(float).values

    lows_idx_all = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5)[1]
    best = None
    best_score = -1e18

    # Search recent candidate event days; prioritize recency
    for i in range(max(20, len(d) - 60), len(d) - 8):
        if i < 1:
            continue
        prev_low = float(L[i - 1]); prev_close = float(C[i - 1])

        strict_gap = float(H[i]) < prev_low
        gap_pct = (float(O[i]) / prev_close - 1.0) if prev_close != 0 else 0.0
        fallback_gap = gap_pct <= -0.03
        if not (strict_gap or fallback_gap):
            continue

        pre0 = max(0, i - 10)
        if i - pre0 < 3:
            continue
        pre_event_high = float(np.nanmax(H[pre0:i]))
        if not np.isfinite(pre_event_high) or pre_event_high <= 0:
            continue

        # Event low within 1-3 days
        j_end = min(len(d), i + 3)
        event_low_idx = int(i + np.nanargmin(L[i:j_end]))
        event_low = float(L[event_low_idx])

        plunge = (pre_event_high - event_low) / pre_event_high
        if plunge < 0.20:
            continue

        # Event volume shock on the event day (i)
        if i < 20:
            continue
        avg20_prior = float(np.nanmean(V[i - 20:i]))
        if not np.isfinite(avg20_prior) or avg20_prior <= 0 or not np.isfinite(V[i]):
            continue
        if float(V[i]) < 1.5 * avg20_prior:
            continue

        # Bounce high after event low
        b_start = event_low_idx + 1
        b_end = min(len(d) - 2, event_low_idx + 20)
        if b_end - b_start < 2:
            continue
        bounce_rel = int(np.nanargmax(H[b_start:b_end + 1]))
        bounce_idx = b_start + bounce_rel
        bounce_high = float(H[bounce_idx])
        if not np.isfinite(bounce_high):
            continue

        decline_amt = pre_event_high - event_low
        if decline_amt <= 0:
            continue
        retr = (bounce_high - event_low) / decline_amt
        if retr < 0.10 or retr > 0.60:
            continue
        if bounce_high >= pre_event_high:
            continue

        # Rollover evidence
        if len(d) - 1 <= bounce_idx + 2:
            continue
        post_bounce_high = float(np.nanmax(H[bounce_idx + 1 :]))
        lower_high = post_bounce_high <= bounce_high * 0.995

        # Break of bounce uptrendline using closes from event_low -> bounce_high
        a_bt, b_bt = _line_fit(np.array([float(event_low_idx), float(bounce_idx)]), np.array([event_low, bounce_high]))
        latest_close = float(C[-1])
        latest_trendline = _line_eval(a_bt, b_bt, float(len(d) - 1))
        trendline_broken = latest_close < latest_trendline

        if not (lower_high or trendline_broken):
            continue

        # Aggressive trigger = lowest swing low after bounce high; conservative trigger = event low
        post_lows = [x for x in lows_idx_all if x > bounce_idx and x < len(d) - 1]
        aggressive_trigger = None
        if post_lows:
            # Lowest swing low after bounce high
            ag_idx = min(post_lows, key=lambda k: float(L[k]))
            aggressive_trigger = (int(ag_idx), float(L[ag_idx]))
        conservative_trigger = (int(event_low_idx), float(event_low))

        if aggressive_trigger is None:
            trig_idx, trig_px = conservative_trigger
            trigger_kind = "conservative_event_low"
        else:
            trig_idx, trig_px = aggressive_trigger
            trigger_kind = "aggressive_post_bounce_low"

        # Must still be in the DCB regime (price below bounce high)
        if latest_close >= bounce_high:
            continue

        recency = i
        score = recency + 5.0 * plunge + 2.0 * (1.0 - abs(retr - 0.33))
        if score > best_score:
            best_score = score
            best = {
                "event_i": int(i),
                "event_low_i": int(event_low_idx),
                "event_low": float(event_low),
                "pre_event_high": float(pre_event_high),
                "bounce_i": int(bounce_idx),
                "bounce_high": float(bounce_high),
                "trigger_i": int(trig_idx),
                "trigger": float(trig_px),
                "trigger_kind": trigger_kind,
                "gap_strict": bool(strict_gap),
                "gap_pct": float(gap_pct),
                "plunge": float(plunge),
                "retr": float(retr),
            }

    if not best:
        return None

    meta: Dict[str, Any] = {
        "annot_type": "dcb",
        "points": [
            _point_meta(d, best["event_i"], float(C[best["event_i"]]), "Event"),
            _point_meta(d, best["event_low_i"], best["event_low"], "Event low"),
            _point_meta(d, best["bounce_i"], best["bounce_high"], "Bounce high"),
            _point_meta(d, best["trigger_i"], best["trigger"], "Trigger"),
        ],
        "lines": [
            _line_meta(d, best["event_low_i"], best["event_low"], best["bounce_i"], best["bounce_high"], "Bounce leg"),
            _line_meta(d, best["event_low_i"], best["event_low"], len(d) - 1, best["event_low"], "Conservative trigger"),
            _line_meta(d, best["trigger_i"], best["trigger"], len(d) - 1, best["trigger"], "Active trigger"),
        ],
        "trigger_kind": best["trigger_kind"],
        "plunge_pct": 100.0 * best["plunge"],
        "bounce_retr_pct": 100.0 * best["retr"],
    }
    return PatternCandidate(pattern="DEAD_CAT_BOUNCE", direction="BREAKDOWN", level=float(best["trigger"]), meta=meta)


def detect_pattern_candidates(df: pd.DataFrame) -> List[PatternCandidate]:
    out: List[PatternCandidate] = []
    hs = detect_hs_top(df)
    if hs:
        out.append(hs)
    ihs = detect_inverse_hs(df)
    if ihs:
        out.append(ihs)
    dcb = detect_dead_cat_bounce(df)
    if dcb:
        out.append(dcb)
    out.extend(detect_structure_candidates(df))
    return out


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

    EARLY is within EARLY_MULT * ATR of the trigger (pre-break 90% zone); if not CONFIRMED, we keep it EARLY.
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
        # "90% there": within 0.5 ATR of trigger (and geometry already validated upstream)
        if abs(close - level) <= EARLY_MULT * atr_val:
            return "EARLY_", dist_atr
        return "", dist_atr

    # BREAKDOWN
    price_ok = close <= level - ATR_CONFIRM_MULT * atr_val
    vol_ok = vol_ratio >= VOL_CONFIRM_MULT
    clv_ok = clv <= CLV_BREAKDOWN_MAX
    if price_ok and vol_ok and clv_ok:
        return "CONFIRMED_", dist_atr
    if abs(close - level) <= EARLY_MULT * atr_val:
        return "EARLY_", dist_atr
    return "", dist_atr


def compute_signals_for_ticker(ticker: str, df: pd.DataFrame) -> List[LevelSignal]:
    sigs: List[LevelSignal] = []
    if df is None or df.empty or len(df) < 80:
        return sigs

    d = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(d) < 80:
        return sigs

    a = atr(d, ATR_N)
    atr_val = float(a.dropna().iloc[-1]) if not a.dropna().empty else float("nan")
    close = float(d["Close"].iloc[-1])
    pct_today = pct_change_last(d)

    # Confirmation gates use volume ratio (vs AvgVol20) and CLV ([-1..+1])
    vol_ratio = 1.0
    if "Volume" in d.columns and not d["Volume"].dropna().empty:
        try:
            v = float(d["Volume"].iloc[-1])
            avg20_prior = float(d["Volume"].iloc[-21:-1].mean()) if len(d) >= 21 else float("nan")
            if not np.isfinite(avg20_prior):
                avg20_prior = float(d["Volume"].tail(20).mean()) if len(d) >= 20 else float("nan")
            if avg20_prior and np.isfinite(avg20_prior) and np.isfinite(v):
                vol_ratio = float(v / avg20_prior)
        except Exception:
            vol_ratio = 1.0

    clv = 0.0
    try:
        hi = float(d["High"].iloc[-1])
        lo = float(d["Low"].iloc[-1])
        if hi > lo:
            clv = (2.0 * close - hi - lo) / (hi - lo)  # CLV in [-1..+1]
            clv = max(-1.0, min(1.0, float(clv)))
    except Exception:
        clv = 0.0

    candidates = detect_pattern_candidates(d)

    # De-duplicate candidates (same pattern/dir/trigger rounded)
    seen = set()
    for cand in candidates:
        key = (cand.pattern, cand.direction, round(float(cand.level), 4))
        if key in seen:
            continue
        seen.add(key)

        prefix, dist_atr = _classify_vs_level(close, cand.level, atr_val, cand.direction, vol_ratio, clv)
        if not prefix:
            continue
        sigs.append(LevelSignal(
            ticker=ticker,
            signal=f"{prefix}{cand.pattern}_{cand.direction}",
            pattern=cand.pattern,
            direction=cand.direction,
            level=float(cand.level),
            close=close,
            atr=atr_val,
            dist_atr=float(dist_atr),
            pct_today=pct_today,
            meta=cand.meta if isinstance(cand.meta, dict) else None,
        ))

    # If a dead-cat-bounce is active, suppress conflicting bullish early signals from triangles/rectangles near the bounce.
    if any(s.pattern == "DEAD_CAT_BOUNCE" for s in sigs):
        filtered: List[LevelSignal] = []
        for s in sigs:
            if s.pattern == "DEAD_CAT_BOUNCE":
                filtered.append(s)
                continue
            if s.direction == "BREAKOUT" and s.signal.startswith("EARLY_"):
                continue
            filtered.append(s)
        sigs = filtered

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

def _annotate_from_signal_meta(ax, sig: LevelSignal) -> bool:
    """Render pattern geometry from deterministic detector metadata. Returns True if used."""
    meta = getattr(sig, "meta", None)
    if not isinstance(meta, dict) or not meta:
        return False

    used = False

    def _to_ts(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    # Draw lines first
    for ln in meta.get("lines", []) or []:
        try:
            t1 = _to_ts(ln.get("t1")); t2 = _to_ts(ln.get("t2"))
            y1 = float(ln.get("y1")); y2 = float(ln.get("y2"))
            if t1 is None or t2 is None or not np.isfinite(y1) or not np.isfinite(y2):
                continue
            ax.plot([t1, t2], [y1, y2], linestyle="--", linewidth=1)
            label = str(ln.get("label") or "")
            if label:
                ax.text(t2, y2, f" {label}", va="bottom")
            used = True
        except Exception:
            continue

    # Draw touch points / pivots
    for pt in meta.get("touch_points", []) or []:
        try:
            t = _to_ts(pt.get("t")); y = float(pt.get("p"))
            if t is None or not np.isfinite(y):
                continue
            ax.scatter([t], [y], s=20)
            used = True
        except Exception:
            continue

    for pt in meta.get("points", []) or []:
        try:
            t = _to_ts(pt.get("t")); y = float(pt.get("p"))
            if t is None or not np.isfinite(y):
                continue
            ax.scatter([t], [y], s=36)
            label = str(pt.get("label") or "")
            if label:
                # Small offset proportional to price
                off = max(abs(y) * 0.01, 0.5)
                if "Event low" in label or label in ("LS", "H", "RS") and sig.pattern == "IHS":
                    ytext = y - off
                else:
                    ytext = y + off
                ax.annotate(label, (t, y), xytext=(t, ytext), textcoords="data",
                            arrowprops=dict(arrowstyle="->", lw=0.8))
            used = True
        except Exception:
            continue

    # Helpful title note for DCB
    if meta.get("annot_type") == "dcb":
        try:
            trig_kind = str(meta.get("trigger_kind", ""))
            if trig_kind:
                ax.text(0.02, 0.10, f"DCB trigger: {trig_kind}", transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.5))
            used = True
        except Exception:
            pass

    return used

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

        used_meta_annotation = _annotate_from_signal_meta(ax, sig)
        if not used_meta_annotation:
            # Fallbacks for legacy signals / older state entries
            if "HS_TOP" in sig.signal or "H&S_TOP" in sig.signal:
                _annotate_hs_top_dt(ax, d.index.to_list(), close, low)
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
    if any(k in sig.signal for k in ["TRIANGLE", "RECT", "BROADEN"]):
        lines.append("- **Structure visual:** chart draws upper/lower boundaries and touch points used for the trigger line.")
    if "HS_TOP" in sig.signal or "IHS" in sig.signal:
        lines.append("- **HS/IHS visual:** chart labels LS/H/RS plus reaction pivots and the neckline trigger.")
    if "DEAD_CAT_BOUNCE" in sig.signal:
        lines.append("- **DCB visual:** chart marks the event day, event low, bounce high and the active breakdown trigger.")
    return "\n".join(lines)
# Reporting utilities
# ----------------------------
def signals_to_df(
    signals: List[LevelSignal],
    category_resolver=None,
    name_resolver=None,
    country_resolver=None,
) -> pd.DataFrame:
    cols = ["Name of Company", "Ticker", "Country", "Signal", "Pattern", "Dir", "Category", "Close", "Level", "Dist(ATR)", "Day%", "Chart"]
    if not signals:
        return pd.DataFrame(columns=cols)
    rows = []
    for s in signals:
        cat = ""
        try:
            if callable(category_resolver):
                cat = str(category_resolver(s.ticker) or "")
        except Exception:
            cat = ""
        name = ""
        country = ""
        try:
            if callable(name_resolver):
                name = str(name_resolver(s.ticker) or "")
        except Exception:
            name = ""
        try:
            if callable(country_resolver):
                country = str(country_resolver(s.ticker) or "")
        except Exception:
            country = ""
        rows.append({
            "Name of Company": name,
            "Ticker": s.ticker,
            "Country": country,
            "Signal": s.signal,
            "Pattern": s.pattern,
            "Dir": s.direction,
            "Category": cat,
            "Close": s.close,
            "Level": s.level,
            "Dist(ATR)": s.dist_atr,
            "Day%": s.pct_today if s.pct_today is not None else np.nan,
            "Chart": s.chart_path or ""
        })
    return pd.DataFrame(rows)



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
    left_cols = {"Name of Company", "Name", "Ticker", "Country", "Category", "Signal", "Pattern", "Dir", "Chart", "Instrument", "Symbol", "symbol"}
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
                        avg20 = float(td2["Volume"].iloc[-21:-1].mean()) if len(td2) >= 21 else np.nan
                        if not np.isfinite(avg20):
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
    watchlist_set = set(custom)

    # Base universe (unchanged behavior for movers/early callouts)
    if args.mode == "full":
        spx = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        base_universe = sorted(set(custom + spx + ndx))
    else:
        base_universe = custom

    if args.max_tickers and args.max_tickers > 0:
        base_universe = base_universe[:args.max_tickers]

    # Optional MSCI World expansion for CONFIRMED technical signals (4B only), driven by local classification CSV.
    msci_df = load_msci_world_classification(MSCI_WORLD_CLASSIFICATION_CSV)
    msci_tickers = [] if msci_df is None or msci_df.empty else [str(x).strip() for x in msci_df["Ticker"].astype(str).tolist()]
    msci_tickers = sorted({t for t in msci_tickers if t and t not in set(base_universe)})
    if msci_tickers:
        print(f"[msci] loaded {len(msci_tickers)} non-watchlist/non-base tickers from {MSCI_WORLD_CLASSIFICATION_CSV}")
    else:
        print(f"[msci] no extra tickers loaded from {MSCI_WORLD_CLASSIFICATION_CSV} (4B remains base universe only)")

    tech_scan_universe = sorted(set(base_universe + msci_tickers))
    category_for_ticker = build_category_resolver(msci_df)
    company_name_for_ticker, country_for_ticker = build_company_country_resolvers(msci_df)

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
    ohlcv = yf_download_chunk(tech_scan_universe)

    # 2) Movers
    # Compute session movers from the watchlist universe (more reliable than scraping Yahoo gainers/losers).
    # With MSCI expansion enabled, the large batch download can occasionally miss a few watchlist names;
    # do a small fallback redownload so >4% watchlist movers are not dropped from section 2 / exec summary.
    session_rows = []
    mover_universe = list(custom)
    missing_for_movers = []
    for t in mover_universe:
        d = ohlcv.get(t)
        if d is None or d.empty or "Close" not in d.columns:
            missing_for_movers.append(t)
            continue
        dd = d.dropna(subset=["Close"])
        if len(dd) < 2:
            missing_for_movers.append(t)
            continue
        c0 = float(dd["Close"].iloc[-2])
        c1 = float(dd["Close"].iloc[-1])
        if c0 == 0 or math.isnan(c0) or math.isnan(c1):
            missing_for_movers.append(t)
            continue
        pct = (c1 / c0 - 1.0) * 100.0
        session_rows.append({"symbol": t, "pct": float(pct)})

    if missing_for_movers:
        try:
            ohlcv_movers_fb = yf_download_chunk(sorted(set(missing_for_movers)))
        except Exception as e:
            print(f"[movers] fallback redownload failed for {len(missing_for_movers)} tickers: {e}")
            ohlcv_movers_fb = {}
        for t in missing_for_movers:
            d = ohlcv_movers_fb.get(t)
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
    wl_set = set(custom)
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

    # Watchlist/base universe: keep EARLY + CONFIRMED (drives 4A + watchlist trend table)
    base_set = set(base_universe)
    for t in base_universe:
        df = ohlcv.get(t)
        if df is None or df.empty:
            continue
        all_signals.extend(compute_signals_for_ticker(t, df))

    # MSCI expansion: CONFIRMED only (4B), no EARLY noise outside the watchlist/base universe
    if msci_tickers:
        for t in msci_tickers:
            df = ohlcv.get(t)
            if df is None or df.empty:
                continue
            sigs = compute_signals_for_ticker(t, df)
            if sigs:
                all_signals.extend([s for s in sigs if s.signal.startswith("CONFIRMED_")])

    early = [s for s in all_signals if s.signal.startswith("EARLY_") and s.ticker in base_set]
    triggered = [s for s in all_signals if s.signal.startswith("CONFIRMED_")]

    def rank_trigger(s: LevelSignal) -> Tuple[int, float]:
        tier = 0
        return (tier, abs(s.dist_atr))

    triggered_sorted = sorted(triggered, key=rank_trigger)
    early_sorted = sorted(early, key=lambda s: abs(s.dist_atr))

    # Charts for confirmed signals (include MSCI names shown in 4B).
    # Use a higher runtime cap so table rows get chart links, while still bounding render time.
    trig_chart_cap = max(int(MAX_CHARTS_TRIGGERED), 260)
    trig_charts = 0
    for s in triggered_sorted:
        if trig_charts >= trig_chart_cap:
            continue
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
        trig_charts += 1

    early_charts = 0
    for s in early_sorted:
        if s.ticker not in base_set:
            continue
        if early_charts >= MAX_CHARTS_EARLY:
            continue
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
        early_charts += 1

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

    df_early = signals_to_df(early_sorted, category_resolver=category_for_ticker, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    df_trig = signals_to_df(triggered_sorted, category_resolver=category_for_ticker, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
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

    md.append("### 4B) Confirmed breakouts / breakdowns (watchlist + MSCI World)\n")
    md.append("_Includes **CONFIRMED** only: close beyond trigger by ≥0.5 ATR AND Volume ≥1.25×AvgVol(20) AND CLV ≥+0.70 (breakout) / ≤−0.70 (breakdown). Categories keep watchlist custom buckets; non-watchlist MSCI names use S&P 500 11-sector labels._ \n")
    md.append("**NEW (today):**\n")
    df_trig_new_tbl = df_trig_new.copy()
    if "Level" in df_trig_new_tbl.columns and "Threshold" not in df_trig_new_tbl.columns:
        df_trig_new_tbl["Threshold"] = df_trig_new_tbl["Level"]
    if not df_trig_new_tbl.empty and "Category" in df_trig_new_tbl.columns:
        df_trig_new_tbl = df_trig_new_tbl.sort_values(["Category", "Signal", "Dist(ATR)"], na_position="last")
    md.append(md_table_from_df(df_trig_new_tbl, cols=["Name of Company", "Ticker", "Country", "Category", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=80))
    md.append("\n**ONGOING:**\n")
    df_trig_old_tbl = df_trig_old.copy()
    if "Level" in df_trig_old_tbl.columns and "Threshold" not in df_trig_old_tbl.columns:
        df_trig_old_tbl["Threshold"] = df_trig_old_tbl["Level"]
    if not df_trig_old_tbl.empty and "Category" in df_trig_old_tbl.columns:
        df_trig_old_tbl = df_trig_old_tbl.sort_values(["Category", "Signal", "Dist(ATR)"], na_position="last")
    md.append(md_table_from_df(df_trig_old_tbl, cols=["Name of Company", "Ticker", "Country", "Category", "Signal", "Close", "Threshold", "Dist(ATR)", "Day%", "Chart"], max_rows=160))
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
    print(f"Universe(base={len(base_universe)}, tech_scan={len(tech_scan_universe)})  Signals: early={len(early_sorted)} triggered={len(triggered_sorted)}")


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
