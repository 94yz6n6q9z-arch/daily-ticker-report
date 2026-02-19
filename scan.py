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
    """
    try:
        xml_text = fetch_url_text(url, timeout=30)
        root = ET.fromstring(xml_text)

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
    return sorted({_clean_ticker(x) for x in read_lines(CUSTOM_TICKERS_PATH)})


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
