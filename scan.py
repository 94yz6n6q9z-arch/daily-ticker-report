#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily Ticker Report (GitHub Pages)
- Market narrative + key headlines (Yahoo Finance)
- Biggest movers: session + after-hours (>= 4%)
- Technical triggers:
    A) Early callouts (~80%): within 0.5 ATR of trigger, not confirmed
    B) Breakout/Breakdown (or about to): soft break (<0.5 ATR) + confirmed (>=0.5 ATR)
- Patterns: H&S / inverse H&S + triangles/wedges/rectangles/broadening (lightweight heuristics)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
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

# Optional local cached constituents (fallback if Wikipedia fetch fails)
SP500_LOCAL = CONFIG_DIR / "universe_sp500.txt"
NDX_LOCAL = CONFIG_DIR / "universe_nasdaq100.txt"


# ----------------------------
# Config knobs (your preferences)
# ----------------------------
MOVER_THRESHOLD_PCT = 4.0

ATR_N = 14
ATR_CONFIRM_MULT = 0.5  # >= 0.5 ATR beyond level = true breakout/breakdown
EARLY_MULT = 0.5        # within 0.5 ATR of level = early callout

LOOKBACK_DAYS = 260 * 2  # ~2y
DOWNLOAD_PERIOD = "3y"
DOWNLOAD_INTERVAL = "1d"
CHUNK_SIZE = 80

MAX_CHARTS_EARLY = 14
MAX_CHARTS_TRIGGERED = 18

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


# ----------------------------
# Helpers: IO
# ----------------------------
def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(ln)
    return lines


def write_text(path: Path, text: str) -> None:
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
# Universe: constituents
# ----------------------------
def _clean_ticker(t: str) -> str:
    # Wikipedia uses BRK.B, BF.B etc -> Yahoo wants BRK-B, BF-B
    t = t.strip()
    t = t.replace(".", "-") if re.fullmatch(r"[A-Z]+\.[A-Z]+", t) else t
    return t


def get_sp500_tickers() -> List[str]:
    # 1) local fallback
    local = read_lines(SP500_LOCAL)
    if local:
        return sorted({_clean_ticker(x) for x in local})

    # 2) Wikipedia
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
        # Usually the table with "Ticker" column exists
        df = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
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
        # remove obvious non-tickers
        tickers = [t for t in tickers if re.fullmatch(r"[\w\-\.\=]+", t)]
        return sorted(set(tickers))
    except Exception:
        return []


def get_custom_tickers() -> List[str]:
    return sorted({_clean_ticker(x) for x in read_lines(CUSTOM_TICKERS_PATH)})


# ----------------------------
# Market data
# ----------------------------
def yf_download_chunk(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Returns dict ticker -> OHLCV df (columns Open High Low Close Volume)
    Uses chunked yf.download for speed.
    """
    out: Dict[str, pd.DataFrame] = {}
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

        # If single ticker, columns are flat
        if isinstance(data.columns, pd.Index):
            if len(chunk) == 1:
                t = chunk[0]
                df = data.dropna(how="all")
                if df.empty:
                    continue
                df = df.rename_axis("Date").reset_index().set_index("Date")
                out[t] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            continue

        # MultiIndex columns: (Field, Ticker) or (Ticker, Field) depending
        # yfinance returns columns like: ('Close', 'AAPL') etc
        if data.columns.nlevels != 2:
            continue

        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        # Determine orientation
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if set(level0) & fields:
            # (Field, Ticker)
            for t in chunk:
                if ("Close", t) not in data.columns:
                    continue
                df = pd.DataFrame({
                    "Open": data[("Open", t)],
                    "High": data[("High", t)],
                    "Low": data[("Low", t)],
                    "Close": data[("Close", t)],
                    "Volume": data[("Volume", t)] if ("Volume", t) in data.columns else np.nan,
                }).dropna(subset=["Close"])
                df.index.name = "Date"
                if not df.empty:
                    out[t] = df
        else:
            # (Ticker, Field)
            for t in chunk:
                if (t, "Close") not in data.columns:
                    continue
                df = pd.DataFrame({
                    "Open": data[(t, "Open")],
                    "High": data[(t, "High")],
                    "Low": data[(t, "Low")],
                    "Close": data[(t, "Close")],
                    "Volume": data[(t, "Volume")] if (t, "Volume") in data.columns else np.nan,
                }).dropna(subset=["Close"])
                df.index.name = "Date"
                if not df.empty:
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
# News + narrative (Yahoo Finance)
# ----------------------------
def fetch_yahoo_headlines(limit: int = 10) -> List[Dict[str, str]]:
    """
    Uses yfinance's news (Yahoo Finance feed) for SPY/QQQ as a proxy for market headlines.
    """
    items: List[Dict[str, str]] = []
    for sym in ["SPY", "QQQ"]:
        try:
            news = yf.Ticker(sym).news or []
            for n in news:
                title = (n.get("title") or "").strip()
                pub = (n.get("publisher") or "").strip()
                link = (n.get("link") or "").strip()
                if title:
                    items.append({"title": title, "publisher": pub, "link": link})
        except Exception:
            continue

    # de-dupe by title
    seen = set()
    uniq = []
    for it in items:
        t = it["title"]
        if t in seen:
            continue
        seen.add(t)
        uniq.append(it)
    return uniq[:limit]


def build_market_narrative(headlines: List[Dict[str, str]]) -> str:
    """
    Lightweight 'theme' summary from headline keywords.
    """
    if not headlines:
        return "Markets: No headline feed available from Yahoo Finance at runtime."

    text = " ".join([h["title"] for h in headlines]).lower()
    themes = []

    def has_any(words):
        return any(w in text for w in words)

    if has_any(["fed", "powell", "minutes", "fomc", "rate", "yields", "treasury", "inflation"]):
        themes.append("Rates/Fed expectations were a key driver (Fed / minutes / yields).")
    if has_any(["ai", "chip", "semiconductor", "nvidia", "software", "cloud"]):
        themes.append("AI positioning stayed central (chips vs. software sensitivity).")
    if has_any(["earnings", "guidance", "forecast", "results"]):
        themes.append("Earnings/guidance headlines moved single names and factor flows.")
    if has_any(["oil", "crude", "wti", "brent", "energy"]):
        themes.append("Energy/oil headlines featured in the tape.")
    if has_any(["china", "tariff", "europe", "ecb", "boj", "geopolitical", "war"]):
        themes.append("Macro/geopolitics influenced risk appetite and rotation.")

    # If nothing detected, keep it generic but useful
    if not themes:
        themes = ["Narrative: mixed cross-currents; watch index leadership + rates for confirmation."]

    return " ".join(themes)


# ----------------------------
# Movers (>=4%)
# - Session movers from Yahoo "markets/stocks/gainers|losers"
# - After-hours movers from Investing.com (fallback: stockanalysis.com)
# ----------------------------
def _read_html_tables(url: str) -> List[pd.DataFrame]:
    import requests
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return pd.read_html(r.text)


def fetch_session_movers_yahoo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (gainers_df, losers_df) with columns: Symbol, Name, % Change, Price (best-effort)
    """
    # Yahoo gainers/losers list (HTML tables)
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
                tables = _read_html_tables(u)
                if not tables:
                    continue
                # pick the largest table
                df = max(tables, key=lambda x: x.shape[0])
                return df
            except Exception:
                continue
        return pd.DataFrame()

    g = pick_table(gain_urls)
    l = pick_table(lose_urls)

    return g, l


def fetch_afterhours_movers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses Investing.com After Hours page (works well for >=4% filtering).
    Returns (gainers, losers).
    """
    # Investing.com provides a clean after-hours table (no login).
    # If it breaks, fallback to stockanalysis afterhours.
    gain = pd.DataFrame()
    lose = pd.DataFrame()

    # Primary: investing.com
    try:
        tables = _read_html_tables("https://www.investing.com/equities/after-hours")
        # This page has multiple tables (most active, gainers, losers). We identify by columns.
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("chg. %" in c or "chg%" in c for c in cols) and any("symbol" in c or "name" in c for c in cols):
                # heuristic: split by sign later
                df = t.copy()
                gain = df
                break
        # If gain is a combined table, we’ll still filter by % sign.
    except Exception:
        pass

    # Fallback: stockanalysis afterhours
    if gain.empty:
        try:
            tables = _read_html_tables("https://stockanalysis.com/markets/afterhours/")
            # First big table is usually gainers; second is losers
            if len(tables) >= 2:
                gain = tables[0]
                lose = tables[1]
        except Exception:
            pass

    # Normalize:
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        # attempt to find a percent column
        pct_col = None
        for c in out.columns:
            if "chg" in str(c).lower() and "%" in str(c).lower():
                pct_col = c
                break
            if "change" in str(c).lower() and "%" in str(c).lower():
                pct_col = c
                break
            if str(c).lower() in ("% change", "change %", "chg %"):
                pct_col = c
                break
        if pct_col is None:
            # try any column containing %
            for c in out.columns:
                if "%" in str(c):
                    pct_col = c
                    break

        if pct_col is not None:
            out["_pct"] = (
                out[pct_col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            out["_pct"] = pd.to_numeric(out["_pct"], errors="coerce")
        else:
            out["_pct"] = np.nan

        # symbol column
        sym_col = None
        for c in out.columns:
            if str(c).lower() in ("symbol", "sym", "ticker"):
                sym_col = c
                break
        if sym_col is None:
            # sometimes "Name" column starts like "AAPL Apple..."
            sym_col = out.columns[0]

        out["_symbol"] = out[sym_col].astype(str).str.split().str[0]
        return out

    if not gain.empty and lose.empty:
        # if investing gave us a single table, split by sign
        ng = normalize(gain)
        gain = ng[ng["_pct"] >= 0].copy()
        lose = ng[ng["_pct"] < 0].copy()
    else:
        gain = normalize(gain)
        lose = normalize(lose)

    return gain, lose


# ----------------------------
# Technical patterns (heuristics, designed to be robust + fast)
# ----------------------------
@dataclass
class LevelSignal:
    ticker: str
    signal: str              # EARLY_..., SOFT_..., CONFIRMED_...
    pattern: str             # e.g., HS_TOP, IHS, TRIANGLE, RECT, WEDGE, BROADEN
    direction: str           # BREAKOUT / BREAKDOWN
    level: float             # trigger price
    close: float
    atr: float
    dist_atr: float          # signed distance in ATR units (close - level)/atr
    pct_today: Optional[float] = None
    chart_path: Optional[str] = None


def _swing_points(series: pd.Series, window: int = 3) -> Tuple[List[int], List[int]]:
    """
    Returns indices of swing highs and swing lows.
    window=3 means local extrema over +/-3 bars.
    """
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
    # y = a*x + b
    if len(x) < 2:
        return (0.0, float(y[-1]) if len(y) else 0.0)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _classify_vs_level(close: float, level: float, atr_val: float, direction: str) -> Tuple[str, float]:
    """
    direction: BREAKOUT means we care about close vs resistance (close above)
               BREAKDOWN means close vs support (close below)
    returns: (prefix EARLY_/SOFT_/CONFIRMED_, dist_atr_signed)
    dist_atr_signed = (close - level)/atr
    """
    if atr_val is None or atr_val <= 0 or math.isnan(atr_val):
        atr_val = max(level * 0.01, 1e-6)

    dist_atr = (close - level) / atr_val

    # For breakout:
    if direction == "BREAKOUT":
        if close >= level + ATR_CONFIRM_MULT * atr_val:
            return "CONFIRMED_", dist_atr
        if close >= level and close < level + ATR_CONFIRM_MULT * atr_val:
            return "SOFT_", dist_atr
        # below level: early if within EARLY_MULT ATR
        if level - close <= EARLY_MULT * atr_val:
            return "EARLY_", dist_atr
        return "", dist_atr

    # For breakdown:
    if close <= level - ATR_CONFIRM_MULT * atr_val:
        return "CONFIRMED_", dist_atr
    if close <= level and close > level - ATR_CONFIRM_MULT * atr_val:
        return "SOFT_", dist_atr
    if close - level <= EARLY_MULT * atr_val:
        return "EARLY_", dist_atr
    return "", dist_atr


def detect_hs_top(df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
    """
    Heuristic Head & Shoulders top:
    - find last 3 swing highs
    - head highest, shoulders similar
    - neckline = avg of swing lows between peaks (simple)
    returns (pattern, direction, neckline_level) or None
    """
    d = df.tail(LOOKBACK_DAYS).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=4)
    if len(highs_idx) < 3 or len(lows_idx) < 2:
        return None

    # take last 8 highs, pick 3 most recent meaningful
    recent_highs = highs_idx[-10:]
    recent_vals = [(i, float(c.iloc[i])) for i in recent_highs]
    # choose last 3 by time (not by value)
    last3 = recent_vals[-3:]
    if len(last3) < 3:
        return None
    (i1, p1), (i2, p2), (i3, p3) = last3

    # head should be middle and highest
    if not (p2 > p1 and p2 > p3):
        return None

    # shoulders similar within ~6%
    if min(p1, p3) <= 0:
        return None
    if abs(p1 - p3) / min(p1, p3) > 0.07:
        return None

    # find lows between i1-i2 and i2-i3
    lows_between_1 = [i for i in lows_idx if i1 < i < i2]
    lows_between_2 = [i for i in lows_idx if i2 < i < i3]
    if not lows_between_1 or not lows_between_2:
        return None
    l1 = float(c.iloc[lows_between_1[-1]])
    l2 = float(c.iloc[lows_between_2[0]])
    neckline = (l1 + l2) / 2.0
    return ("HS_TOP", "BREAKDOWN", neckline)


def detect_inverse_hs(df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
    """
    Heuristic Inverse H&S:
    - last 3 swing lows
    - head lowest in middle
    - neckline from highs between troughs
    returns (pattern, direction, neckline_level) or None
    """
    d = df.tail(LOOKBACK_DAYS).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=4)
    if len(lows_idx) < 3 or len(highs_idx) < 2:
        return None

    recent_lows = lows_idx[-10:]
    recent_vals = [(i, float(c.iloc[i])) for i in recent_lows]
    last3 = recent_vals[-3:]
    if len(last3) < 3:
        return None
    (i1, p1), (i2, p2), (i3, p3) = last3

    if not (p2 < p1 and p2 < p3):
        return None

    # shoulders similar within ~7%
    if min(p1, p3) <= 0:
        return None
    if abs(p1 - p3) / min(p1, p3) > 0.08:
        return None

    # highs between i1-i2 and i2-i3
    highs_between_1 = [i for i in highs_idx if i1 < i < i2]
    highs_between_2 = [i for i in highs_idx if i2 < i < i3]
    if not highs_between_1 or not highs_between_2:
        return None
    h1 = float(c.iloc[highs_between_1[-1]])
    h2 = float(c.iloc[highs_between_2[0]])
    neckline = (h1 + h2) / 2.0
    return ("IHS", "BREAKOUT", neckline)


def detect_triangle_wedge_rect_broadening(df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
    """
    Very lightweight structure detector:
    - Fit trendlines to recent swing highs/lows; classify by slopes.
    - Trigger level for breakout/breakdown: nearer boundary (upper for breakout, lower for breakdown)
    returns (pattern, direction, level)
    """
    d = df.tail(180).copy()
    c = d["Close"].dropna()
    if len(c) < 120:
        return None

    highs_idx, lows_idx = _swing_points(c, window=3)
    if len(highs_idx) < 4 or len(lows_idx) < 4:
        return None

    # Use last 6 swing highs/lows
    hi = np.array(highs_idx[-6:], dtype=int)
    lo = np.array(lows_idx[-6:], dtype=int)

    xh = hi.astype(float)
    yh = np.array([float(c.iloc[i]) for i in hi])
    xl = lo.astype(float)
    yl = np.array([float(c.iloc[i]) for i in lo])

    a_u, b_u = _line_fit(xh, yh)  # upper trendline
    a_l, b_l = _line_fit(xl, yl)  # lower trendline

    # Evaluate at last bar
    x_last = float(len(c) - 1)
    upper_now = a_u * x_last + b_u
    lower_now = a_l * x_last + b_l

    # Basic sanity:
    if not (upper_now > lower_now):
        return None

    # Converging / diverging width:
    x_early = float(max(0, len(c) - 60))
    width_now = (a_u * x_last + b_u) - (a_l * x_last + b_l)
    width_then = (a_u * x_early + b_u) - (a_l * x_early + b_l)
    if width_then <= 0:
        return None

    converging = width_now < 0.75 * width_then
    diverging = width_now > 1.15 * width_then

    # Rectangle: slopes near 0, stable range
    if abs(a_u) < 0.02 and abs(a_l) < 0.02 and (width_now / max(lower_now, 1e-6)) < 0.18:
        # trigger on both sides; we will return the nearer side later
        # Here choose both? We'll return breakout level by default.
        return ("RECT", "BOTH", float(upper_now))

    if diverging:
        # Broadening (megaphone)
        return ("BROADEN", "BOTH", float(upper_now))

    if converging:
        # triangle vs wedge
        if a_u < 0 and a_l > 0:
            return ("TRIANGLE", "BOTH", float(upper_now))
        if a_u < 0 and a_l < 0:
            return ("WEDGE_DOWN", "BOTH", float(upper_now))
        if a_u > 0 and a_l > 0:
            return ("WEDGE_UP", "BOTH", float(upper_now))
        # fallback
        return ("TRIANGLE", "BOTH", float(upper_now))

    return None


def compute_signals_for_ticker(ticker: str, df: pd.DataFrame) -> List[LevelSignal]:
    sigs: List[LevelSignal] = []
    if df is None or df.empty or len(df) < 80:
        return sigs

    d = df.copy()
    d = d.dropna(subset=["Close"])
    if len(d) < 80:
        return sigs

    a = atr(d, ATR_N)
    atr_val = float(a.dropna().iloc[-1]) if not a.dropna().empty else float("nan")
    close = float(d["Close"].iloc[-1])
    pct_today = pct_change_last(d)

    # 1) H&S top
    hs = detect_hs_top(d)
    if hs:
        pattern, direction, level = hs
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction)
        if prefix:
            sigs.append(LevelSignal(
                ticker=ticker,
                signal=f"{prefix}{pattern}_{direction}",
                pattern=pattern,
                direction=direction,
                level=float(level),
                close=close,
                atr=atr_val,
                dist_atr=float(dist_atr),
                pct_today=pct_today
            ))

    # 2) Inverse H&S
    ihs = detect_inverse_hs(d)
    if ihs:
        pattern, direction, level = ihs
        prefix, dist_atr = _classify_vs_level(close, level, atr_val, direction)
        if prefix:
            sigs.append(LevelSignal(
                ticker=ticker,
                signal=f"{prefix}{pattern}_{direction}",
                pattern=pattern,
                direction=direction,
                level=float(level),
                close=close,
                atr=atr_val,
                dist_atr=float(dist_atr),
                pct_today=pct_today
            ))

    # 3) Structure (triangle/wedge/rect/broaden)
    st = detect_triangle_wedge_rect_broadening(d)
    if st:
        pattern, direction, upper_level = st
        # For BOTH patterns, decide which side is relevant based on where price is
        # Create two potential triggers: breakout at upper, breakdown at lower (if we can estimate)
        # We estimate lower via recent swing lows median.
        c = d.tail(180)["Close"].dropna()
        _, lows_idx = _swing_points(c, window=3)
        lower_level = float(np.median([float(c.iloc[i]) for i in lows_idx[-6:]])) if len(lows_idx) >= 6 else float(c.min())

        # Breakout check
        prefix_u, dist_u = _classify_vs_level(close, float(upper_level), atr_val, "BREAKOUT")
        if prefix_u:
            sigs.append(LevelSignal(
                ticker=ticker,
                signal=f"{prefix_u}{pattern}_BREAKOUT",
                pattern=pattern,
                direction="BREAKOUT",
                level=float(upper_level),
                close=close,
                atr=atr_val,
                dist_atr=float(dist_u),
                pct_today=pct_today
            ))
        # Breakdown check
        prefix_l, dist_l = _classify_vs_level(close, float(lower_level), atr_val, "BREAKDOWN")
        if prefix_l:
            sigs.append(LevelSignal(
                ticker=ticker,
                signal=f"{prefix_l}{pattern}_BREAKDOWN",
                pattern=pattern,
                direction="BREAKDOWN",
                level=float(lower_level),
                close=close,
                atr=atr_val,
                dist_atr=float(dist_l),
                pct_today=pct_today
            ))

    return sigs


# ----------------------------
# Charting
# ----------------------------
def plot_signal_chart(ticker: str, df: pd.DataFrame, sig: LevelSignal) -> Optional[str]:
    """
    Saves chart to docs/img and returns relative path.
    """
    if df is None or df.empty:
        return None

    d = df.tail(220).copy()
    d = d.dropna(subset=["Close"])
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

    # annotate last close
    ax.scatter([d.index[-1]], [d["Close"].iloc[-1]])
    ax.text(d.index[-1], d["Close"].iloc[-1], f"  {d['Close'].iloc[-1]:.2f}", va="center")

    fname = f"{ticker.replace('/', '_')}_{sig.signal}.png"
    fname = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", fname)
    out_path = IMG_DIR / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return f"img/{fname}"


# ----------------------------
# Report building
# ----------------------------
def format_snapshot_table(rows: List[Tuple[str, float, float, float]]) -> str:
    # rows: (Name, last, pct, prev)
    md = []
    md.append("| Instrument | Last | Day % | Prev |")
    md.append("|---|---:|---:|---:|")
    for name, last, pct, prev in rows:
        md.append(f"| {name} | {last:.2f} | {pct:+.2f}% | {prev:.2f} |")
    return "\n".join(md)


def fetch_market_snapshot() -> List[Tuple[str, float, float, float]]:
    instruments = {
        "Nasdaq 100 (^NDX)": "^NDX",
        "S&P 500 (^GSPC)": "^GSPC",
        "QQQ": "QQQ",
        "SPY": "SPY",
        "VIX (^VIX)": "^VIX",
        "US 10Y (^TNX)": "^TNX",
        "DXY (DX-Y.NYB)": "DX-Y.NYB",
        "WTI (CL=F)": "CL=F",
    }
    rows = []
    try:
        data = yf.download(
            tickers=list(instruments.values()),
            period="7d",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return rows

    for label, sym in instruments.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[("Close", sym)].dropna()
            else:
                # single column case (rare)
                close = data["Close"].dropna()

            if len(close) < 2:
                continue
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            pct = (last / prev - 1.0) * 100.0
            rows.append((label, last, pct, prev))
        except Exception:
            continue
    return rows


def signals_to_df(signals: List[LevelSignal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(columns=["Ticker", "Signal", "Pattern", "Dir", "Close", "Level", "Dist(ATR)", "Day%"])
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
    # nicer formatting
    for c in ["Close", "Level"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "Dist(ATR)" in d.columns:
        d["Dist(ATR)"] = pd.to_numeric(d["Dist(ATR)"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    if "Day%" in d.columns:
        d["Day%"] = pd.to_numeric(d["Day%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "Chart" in d.columns:
        # convert to markdown link
        d["Chart"] = d["Chart"].apply(lambda p: f"[chart]({p})" if isinstance(p, str) and p else "")

    return d[cols].to_markdown(index=False)


def diff_new_ended(prev: Dict[str, List[str]], cur: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    prev_set = set(prev.get("signals", []))
    cur_set = set(cur.get("signals", []))
    new = sorted(cur_set - prev_set)
    ended = sorted(prev_set - cur_set)
    return new, ended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "custom"], default=os.environ.get("MODE", "full"))
    ap.add_argument("--max-tickers", type=int, default=int(os.environ.get("MAX_TICKERS", "0")))
    args = ap.parse_args()

    # Universe
    custom = get_custom_tickers()
    spx = get_sp500_tickers()
    ndx = get_nasdaq100_tickers()

    if args.mode == "custom":
        universe = custom
    else:
        universe = sorted(set(custom + spx + ndx))

    if args.max_tickers and args.max_tickers > 0:
        universe = universe[:args.max_tickers]

    now = dt.datetime.now(dt.timezone.utc)
    ts_local = now.astimezone()  # runner local tz
    header_time = ts_local.strftime("%Y-%m-%d %H:%M %Z")

    # Market overview
    snapshot_rows = fetch_market_snapshot()
    headlines = fetch_yahoo_headlines(limit=10)
    narrative = build_market_narrative(headlines)

    # Movers (>=4%)
    session_g, session_l = fetch_session_movers_yahoo()
    ah_g, ah_l = fetch_afterhours_movers()

    def filter_movers(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        # best-effort % column detection
        pct_col = None
        for c in out.columns:
            if "%" in str(c) and ("change" in str(c).lower() or "chg" in str(c).lower()):
                pct_col = c
                break
        if pct_col is None and "_pct" in out.columns:
            out["pct"] = out["_pct"]
        else:
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

        # symbol column
        sym_col = None
        for c in out.columns:
            if str(c).lower() in ("symbol", "ticker"):
                sym_col = c
                break
        if sym_col is None and "_symbol" in out.columns:
            out["symbol"] = out["_symbol"]
        else:
            out["symbol"] = out[sym_col].astype(str) if sym_col else out.iloc[:, 0].astype(str)

        out = out.dropna(subset=["pct"])
        out = out.loc[out["pct"].abs() >= MOVER_THRESHOLD_PCT].copy()
        out = out.sort_values("pct", ascending=False)
        return out[["symbol", "pct"]].head(30)

    session_gf = filter_movers(session_g)
    session_lf = filter_movers(session_l).sort_values("pct", ascending=True)
    ah_gf = filter_movers(ah_g)
    ah_lf = filter_movers(ah_l).sort_values("pct", ascending=True)

    # Download OHLCV once
    ohlcv = yf_download_chunk(universe)

    # Compute signals
    all_signals: List[LevelSignal] = []
    for t in universe:
        df = ohlcv.get(t)
        if df is None or df.empty:
            continue
        sigs = compute_signals_for_ticker(t, df)
        all_signals.extend(sigs)

    # Categorize
    early = [s for s in all_signals if s.signal.startswith("EARLY_")]
    triggered = [s for s in all_signals if s.signal.startswith("SOFT_") or s.signal.startswith("CONFIRMED_")]

    # Chart only top N (prioritize confirmed, then soft, then early by |dist|)
    def rank_key(s: LevelSignal) -> Tuple[int, float]:
        # confirmed first
        tier = 0 if s.signal.startswith("CONFIRMED_") else (1 if s.signal.startswith("SOFT_") else 2)
        return (tier, abs(s.dist_atr))

    triggered_sorted = sorted(triggered, key=rank_key)
    early_sorted = sorted(early, key=lambda s: abs(s.dist_atr))

    # Create charts
    for s in triggered_sorted[:MAX_CHARTS_TRIGGERED]:
        p = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
        s.chart_path = p
    for s in early_sorted[:MAX_CHARTS_EARLY]:
        p = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s)
        s.chart_path = p

    # Build NEW/ENDED diff
    state = load_state()
    prev = {"signals": state.get("signals", [])}
    cur_ids = [f"{s.ticker}|{s.signal}" for s in all_signals]
    state["signals"] = cur_ids
    save_state(state)
    new_ids, ended_ids = diff_new_ended(prev, {"signals": cur_ids})

    # Convert to dfs
    df_early = signals_to_df(early_sorted)
    df_trig = signals_to_df(triggered_sorted)

    # Split NEW vs ONGOING for readability
    new_set = set(new_ids)

    def mark_new(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            return df, df
        d = df.copy()
        d["_id"] = d["Ticker"].astype(str) + "|" + d["Signal"].astype(str)
        d_new = d[d["_id"].isin(new_set)].drop(columns=["_id"])
        d_old = d[~d["_id"].isin(new_set)].drop(columns=["_id"])
        return d_new, d_old

    df_early_new, df_early_old = mark_new(df_early)
    df_trig_new, df_trig_old = mark_new(df_trig)

    # Markdown assembly
    md = []
    md.append(f"# Daily Report\n")
    md.append(f"_Generated: **{header_time}**_\n")

    # 1) Market narrative
    md.append("## 1) Market recap & positioning\n")
    md.append(f"{narrative}\n")

    if snapshot_rows:
        md.append("**Key tape (daily %):**\n")
        md.append(format_snapshot_table(snapshot_rows))
        md.append("")

    if headlines:
        md.append("**Yahoo Finance headlines (context for the narrative):**\n")
        for h in headlines[:8]:
            # link is optional; keep it clean
            title = h["title"]
            pub = h["publisher"]
            link = h["link"]
            if link:
                md.append(f"- [{title}]({link}) — {pub}".rstrip())
            else:
                md.append(f"- {title} — {pub}".rstrip())
        md.append("")

    # 2) Movers
    md.append("## 2) Biggest movers (≥ 4%)\n")

    def movers_table(df: pd.DataFrame, title: str) -> str:
        if df is None or df.empty:
            return f"**{title}:** _None ≥ {MOVER_THRESHOLD_PCT:.0f}%_\n"
        t = df.copy()
        t["pct"] = t["pct"].map(lambda x: f"{x:+.2f}%")
        return f"**{title}:**\n\n" + t.to_markdown(index=False) + "\n"

    md.append(movers_table(session_gf, "Session gainers"))
    md.append(movers_table(session_lf, "Session losers"))
    md.append(movers_table(ah_gf, "After-hours gainers"))
    md.append(movers_table(ah_lf, "After-hours losers"))

    # 4) Technical triggers (split)
    md.append("## 4) Technical triggers\n")
    md.append(f"**Breakout confirmation rule:** close beyond trigger by **≥ {ATR_CONFIRM_MULT:.1f} ATR**.\n")

    md.append("### 4A) Early callouts (~80% complete)\n")
    md.append("_These are close enough to the trigger that you should pre-plan the trade. “Close enough” = within 0.5 ATR of the neckline/boundary._\n")

    md.append("**NEW (today):**\n")
    md.append(md_table_from_df(df_early_new, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=30))
    md.append("\n**ONGOING:**\n")
    md.append(md_table_from_df(df_early_old, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=30))
    md.append("")

    md.append("### 4B) Breakouts / breakdowns (or about to)\n")
    md.append("_Includes **SOFT** breaks (pierced the level but <0.5 ATR) and **CONFIRMED** breaks (≥0.5 ATR)._ \n")

    md.append("**NEW (today):**\n")
    md.append(md_table_from_df(df_trig_new, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=40))
    md.append("\n**ONGOING:**\n")
    md.append(md_table_from_df(df_trig_old, cols=["Ticker", "Signal", "Close", "Level", "Dist(ATR)", "Day%", "Chart"], max_rows=40))
    md.append("")

    # 5) Needle-moving catalysts (Yahoo Finance)
    md.append("## 5) Needle-moving catalysts to watch (Yahoo Finance)\n")
    md.append("_Interpretation: focus on the items most likely to move risk by 4–5%+ (rates, AI positioning, major earnings/guidance, macro prints)._ \n")
    if headlines:
        for h in headlines[:10]:
            title = h["title"]
            pub = h["publisher"]
            link = h["link"]
            if link:
                md.append(f"- [{title}]({link}) — {pub}".rstrip())
            else:
                md.append(f"- {title} — {pub}".rstrip())
    else:
        md.append("_No Yahoo Finance headlines available._")
    md.append("")

    # Ended signals
    md.append("## Changelog\n")
    if new_ids:
        md.append("**New signals:**\n")
        for x in new_ids[:60]:
            md.append(f"- {x}")
    else:
        md.append("**New signals:** _None_\n")

    if ended_ids:
        md.append("\n**Ended signals:**\n")
        for x in ended_ids[:60]:
            md.append(f"- {x}")
    else:
        md.append("\n**Ended signals:** _None_\n")

    md_text = "\n".join(md).strip() + "\n"

    # Write both report.md and index.md so Pages root shows it
    write_text(REPORT_PATH, md_text)
    write_text(INDEX_PATH, md_text)

    print(f"Wrote: {REPORT_PATH}")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Signals: early={len(early_sorted)} triggered={len(triggered_sorted)} universe={len(universe)}")


if __name__ == "__main__":
    main()
