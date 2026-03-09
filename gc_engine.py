#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Growth Compounder Engine (gc_engine.py)
=======================================
Standalone module for detecting multi-year growth compounders.
Designed to run independently during development/backtesting,
then merge into scan.py Section 7 once polished.

Step 1: Data Layer
- Download earnings dates, quarterly revenue, EPS surprise for universe
- Compute YoY revenue growth, acceleration, sector medians
- Cache in gc_state.json

Usage:
    python gc_engine.py --mode data       # Download + cache earnings data
    python gc_engine.py --mode scan       # Run ignition detection (Step 2+)
    python gc_engine.py --mode backtest   # Run 20-year backtest (Step 6)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
GC_VERSION = "0.5.0"

# Version history (mirrors the changelog pattern in scan.py)
_GC_VERSION_LOG: dict = {
    "0.1.0": (
        "Initial release. Data layer: earnings dates, quarterly revenue, EPS surprise "
        "for MSCI World universe. Compute YoY revenue growth, acceleration, sector medians. "
        "Cache in gc_state.json."
    ),
    "0.2.0": (
        "v91 integration: MSCI EM universe merged into load_universe() via msci_em_classification.csv. "
        "TICKER_OVERRIDES dict auto-corrects 7 known bad mappings. "
        "compute_revenue_analytics() falls back to info.revenue_growth for markets with no "
        "quarterly data (recovers ~348 tickers: Japan, UK, Australia, France, Switzerland). "
        "is_ghost_ticker() filter added to skip Bloomberg placeholders before any yfinance call. "
        "FMP_TARGET_EXCHANGES list for exchanges where yfinance is structurally weak. "
        "Batch scheduling: US tickers daily (Mon–Fri); RoW tickers once/week via stable hash. "
        "EPS_BEAT_STREAK_MIN lowered from 3 to 2."
    ),
    "0.3.0": (
        "v92 sync: universe label updated to 'MSCI World + EM' in all user-facing log output. "
        "GC_VERSION constant now follows scan.py version-log pattern so both files track "
        "changes identically. No logic changes — version bump is documentation only."
    ),
    "0.5.0": (
        "Performance + correctness pass. "
        "Method 2 (tk.quarterly_earnings / tk.earnings) removed — deprecated in yfinance 1.2, "
        "was source of DeprecationWarning on every ticker. "
        "KNOWN_DEAD_TICKERS set added: 32 sanctioned Russian + Gulf tickers permanently "
        "skipped before any API call (~3 min savings per force-reload). "
        "Sleep reductions: every-100 pause 1.0→0.75s, per-5 US 0.30→0.22s, "
        "RoW 0.15→0.11s, retry entry 5.0→3.5s, retry per-ticker 0.40→0.30s. "
        "EPS match zone: ±$0.01 returns 'match' (streak-neutral) instead of miss. "
        "Revenue match zone: ±0.5% consensus returns 'match' (streak-neutral). "
        "_revenue_beat_for_row YoY proxy removed — returns None when no consensus estimate "
        "(YoY growth ≠ consensus beat; prior behavior was misleading). "
        "latest_eps_result / latest_rev_result fields added: 'beat'/'match'/'miss'/'unknown'. "
        "FMP enrichment: tries api/v3 fallback URL after stable endpoint. "
        "_fmp_enrich_filled key renamed to _fmp_rev_estimates_filled (summary counter fix). "
        "Yahoo quoteSummary revenue: documented as structurally returning 0 — "
        "earningsHistory has no revenue fields; earningsTrend is forward-only."
    ),
    "0.4.0": (
        "Layer star-gate redesign. Star 1: technical ignition (unchanged). "
        "Star 2: BOTH eps_beat_streak>=2 AND revenue_beat_streak>=2 (YoY proxy when estimates unavailable) OR a massive catalyst "
        "confirmed by OpenAI (not just keyword match). Star 3: Star 2 + rev>=20% YoY + "
        "OpenAI moat confirmation. compute_eps_analytics() gains revenue_beat_streak. "
        "_openai_catalyst_is_massive() added: sends headline to gpt-4o-mini to judge whether "
        "it is a genuinely company-thesis-changing event. _openai_moat_assessment() added: "
        "sends company profile to gpt-4o-mini to confirm durable moat. Both fall back "
        "gracefully when OPENAI_API_KEY is absent. Star counts logged in data summary."
    ),
}

BASE_DIR = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parent))
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR = BASE_DIR / "docs"
GC_STATE_PATH = DOCS_DIR / "gc_state.json"
MSCI_CSV = CONFIG_DIR / "msci_world_classification.csv"
MSCI_EM_CSV = CONFIG_DIR / "msci_em_classification.csv"

# OHLCV download
DOWNLOAD_PERIOD = "3y"        # For daily scanning
DOWNLOAD_PERIOD_BACKTEST = "max"  # For 20-year backtest
DOWNLOAD_INTERVAL = "1d"
CHUNK_SIZE = 80

# ATR
ATR_N = 14

# Earnings data cache TTL (don't re-download if fresher than this)
EARNINGS_CACHE_TTL_HOURS = 20  # Re-download once per day

# ────────────────────────────────────────────────────────────────
# Ticker overrides: bad/unmapped symbol → correct Yahoo Finance symbol
# Add here rather than editing the CSV manually — gc_engine auto-heals
# even if update_msci_world_classification.py regenerates the CSV.
# ────────────────────────────────────────────────────────────────
TICKER_OVERRIDES: Dict[str, str] = {
    # Runtime-only corrections — applied when gc_engine reads the CSV at fetch time.
    # If a correction can be determined from the raw iShares export (company name,
    # exchange, country), it belongs in update_msci_world_classification.py instead.
    # Only keep here what genuinely cannot be known until Yahoo Finance is queried.
    # (currently empty — all known overrides moved to update_msci_world_classification.py)
}

# ────────────────────────────────────────────────────────────────
# Ghost ticker filter — drop from universe before any yfinance call
# Bloomberg placeholder pattern: purely numeric (no dot), ends in D,
# or known junk strings that appear in MSCI EM CSV exports.
# ────────────────────────────────────────────────────────────────
import re as _re
_GHOST_PATTERN = _re.compile(
    r"^-$"                # bare dash
    r"|^\d+D?$"           # pure numeric, optionally ending in D
    r"|^[A-Z]{1,5}\d{6,}D$"  # alpha prefix + long numeric + D
    r"|^\.$"              # bare dot
    r"|^DUMMY$"           # iShares CSV placeholder row
)


# ────────────────────────────────────────────────────────────────
# Known-dead tickers — permanently skip before any API call.
# These symbols exist in the MSCI CSV (often as legacy/sanctioned
# holdings) but have NO Yahoo Finance support and return 404 on
# every call, wasting ~4-6 seconds each per run.
#
# Categories:
#   - Sanctioned Russian equities: will never be re-listed on Yahoo
#   - Gulf tickers with no Yahoo quoteSummary support (DHBK, DUBK etc.)
#   - EM tickers confirmed dead after 90d of 404s
#
# To add: append ticker + comment explaining why.
# Do NOT put correctable symbols here — use TICKER_OVERRIDES instead.
# ────────────────────────────────────────────────────────────────
KNOWN_DEAD_TICKERS: set = {
    # Sanctioned Russian equities — permanently 404 on Yahoo Finance
    "AFKS", "AFLT", "BLDN", "BRES", "CBOM", "CBQK", "CHMF",
    "FEES", "FLOT", "GAZP", "GEMC", "GISS", "GMKN", "IRAO",
    "LKOH", "LSRG", "MGNT", "NLMK", "NVTK", "PHOR", "PIKK",
    "RTKM", "RUAL", "SBER", "SGZH", "SNGS", "SNGSP", "TATN",
    "TCSG", "UPRO", "VKCO",
    # Gulf tickers with no Yahoo Finance support
    "DHBK", "DUBK", "IGRD", "IQCD", "UDCD", "VFQS",
}


def is_ghost_ticker(ticker: str) -> bool:
    """Return True for any symbol that will never resolve on Yahoo Finance.

    The MSCI CSV is already cleaned by update_msci_world_classification.py —
    this is a lightweight safety net for stale cache entries and edge cases
    that slip through (e.g. a ticker that was valid when cached but has since
    become a Bloomberg placeholder after a corporate action).
    """
    t = str(ticker).strip()
    if not t:
        return True
    if "." not in t and t.replace("-", "").isdigit():
        return True
    if t.count(".") > 1:   # multi-dot = malformed (e.g. BAJAJ.AUTO.NS)
        return True
    return bool(_GHOST_PATTERN.match(t))

# ────────────────────────────────────────────────────────────────
# Star 2 threshold — consecutive EPS beats required
# Lowered from 3 to 2 per spec revision 2026-03-08
# ────────────────────────────────────────────────────────────────
EPS_BEAT_STREAK_MIN = 2

# ────────────────────────────────────────────────────────────────
# Batch scheduling
# US tickers: fetched every weekday (Mon–Fri)
# RoW tickers: assigned a stable day-of-week (0–6) via ticker hash
#              so each ticker is refreshed once per week, any day
# Earnings trigger: any ticker with earnings_date within 1 day is
#                   force-fetched regardless of batch assignment
# ────────────────────────────────────────────────────────────────
def assign_batch_day(ticker: str) -> int:
    """Stable day-of-week (0=Mon … 6=Sun) for RoW tickers, based on hash."""
    return int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 7


def _has_earnings_today(cached: Dict[str, Any], today: dt.date) -> bool:
    """Return True if any earnings date in cache falls within 1 day of today."""
    for ed in cached.get("earnings_dates", []):
        try:
            ed_date = dt.date.fromisoformat(str(ed.get("date", ""))[:10])
            if abs((today - ed_date).days) <= 1:
                return True
        except Exception:
            pass
    return False


def _should_fetch_today(ticker: str, cached: Dict[str, Any], now: dt.datetime, force: bool) -> bool:
    """Decide whether this ticker should be fetched in today's run.

    Rules (in priority order):
      1. force=True              → always fetch
      2. inactive constituent    → never fetch (left MSCI universe)
      3. new_constituent         → always fetch immediately (just joined MSCI)
      4. earnings trigger        → fetch if earnings ±1 day
      5. US ticker               → fetch on weekdays only
      6. RoW ticker              → fetch on assigned batch day (any day of week)
    """
    if force:
        return True
    if cached.get("inactive"):
        return False
    if cached.get("new_constituent"):
        return True
    today = now.date()
    if _has_earnings_today(cached, today):
        return True
    exch = ticker.rsplit(".", 1)[-1] if "." in ticker else "US"
    if exch == "US":
        return now.weekday() < 5   # Mon–Fri
    batch_day = cached.get("batch_day", assign_batch_day(ticker))
    return now.weekday() == batch_day


# ────────────────────────────────────────────────────────────────
# FMP target exchanges — yfinance structurally weak here.
# These exchanges get FMP fallback first before being marked empty.
# Ordered by coverage gap severity (worst first).
# ────────────────────────────────────────────────────────────────
FMP_TARGET_EXCHANGES = ["KL", "IS", "PS", "AD", "DU", "T", "AX", "L", "JO", "HK", "PA", "SW"]

# ────────────────────────────────────────────────────────────────
# Catalyst keywords for major event detection via yfinance news
# Events that qualify as Layer-2 equivalent WITHOUT earnings beats
# Must be genuinely market-moving at the company/thesis level.
# ────────────────────────────────────────────────────────────────
CATALYST_KEYWORDS_TIER1 = [
    # Regulatory / FDA
    "fda approv", "fda clears", "fda grants", "breakthrough therapy",
    "full approval", "accelerated approval", "pdufa", "nda approv", "bla approv",
    # Major contracts / geopolitical uplift
    "awarded contract", "major contract", "multi-billion", "multi-year contract",
    "defense contract", "nato contract", "government contract",
    # Product launches that reshape a market
    "launches", "commercial launch", "product launch", "enters market",
    # M&A as catalyst
    "acquisition", "merger", "takeover bid", "buyout",
    # Commodity / geopolitical shocks
    "sanctions", "export ban", "supply shock", "opec",
]
CATALYST_KEYWORDS_TIER2 = [
    "partnership", "strategic alliance", "licensing agreement",
    "record revenue", "record earnings", "raised guidance", "guidance raised",
    "beats estimates", "beat expectations", "exceeds forecast",
]


# ────────────────────────────────────────────────────────────────
# Utilities (shared with scan.py — will import from scan.py on merge)
# ────────────────────────────────────────────────────────────────
def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


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


def clv_at_bar(d: pd.DataFrame, i: int) -> float:
    try:
        h = float(d["High"].iloc[i])
        l = float(d["Low"].iloc[i])
        c = float(d["Close"].iloc[i])
        rng = h - l
        if rng <= 0:
            return 0.0
        return float(((c - l) - (h - c)) / rng)
    except Exception:
        return float("nan")


# ────────────────────────────────────────────────────────────────
# State management
# ────────────────────────────────────────────────────────────────
def load_gc_state() -> Dict[str, Any]:
    if GC_STATE_PATH.exists():
        try:
            return json.loads(GC_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_gc_state(state: Dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    GC_STATE_PATH.write_text(json.dumps(state, default=_json_default, indent=1), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ────────────────────────────────────────────────────────────────
# Universe construction
# ────────────────────────────────────────────────────────────────
def load_universe() -> pd.DataFrame:
    """Load MSCI World + MSCI EM classification CSVs.
    Returns combined DataFrame with columns: Ticker, Company, Country, Sector
    Deduplicates by Ticker — World takes priority over EM for any overlap.
    """
    frames = []

    # MSCI World (primary)
    if not MSCI_CSV.exists():
        print(f"[gc] MSCI World CSV not found at {MSCI_CSV}")
    else:
        try:
            df = pd.read_csv(MSCI_CSV, dtype=str)
            df["Ticker"] = df["Ticker"].astype(str).str.strip()
            df = df[df["Ticker"].str.len() > 0]
            df["_source"] = "world"
            frames.append(df)
            print(f"[gc] Loaded {len(df)} tickers from MSCI World CSV")
        except Exception as e:
            print(f"[gc] Error loading MSCI World CSV: {e}")

    # MSCI EM (supplementary)
    if not MSCI_EM_CSV.exists():
        print(f"[gc] MSCI EM CSV not found at {MSCI_EM_CSV} — universe is World-only. "
              f"Run update_msci_world_classification.py --universe em to generate.")
    else:
        try:
            df_em = pd.read_csv(MSCI_EM_CSV, dtype=str)
            df_em["Ticker"] = df_em["Ticker"].astype(str).str.strip()
            df_em = df_em[df_em["Ticker"].str.len() > 0]
            df_em["_source"] = "em"
            frames.append(df_em)
            print(f"[gc] Loaded {len(df_em)} tickers from MSCI EM CSV")
        except Exception as e:
            print(f"[gc] Error loading MSCI EM CSV: {e}")

    if not frames:
        return pd.DataFrame(columns=["Ticker", "Company", "Country", "Sector"])

    combined = pd.concat(frames, ignore_index=True)
    # World takes priority for duplicates
    combined = combined.sort_values("_source", ascending=True)  # 'em' > 'world' alphabetically — world sorts first
    combined = combined.drop_duplicates(subset=["Ticker"], keep="first")
    combined = combined.drop(columns=["_source"], errors="ignore")

    # Normalize exchange-specific ticker format quirks before any lookup:
    # Borsa Istanbul (BIST): iShares/MSCI carry the equity share class suffix "-E"
    # (e.g. KCHOL-E.IS, BIMAS-E.IS) which yfinance does NOT recognise — it expects
    # plain KCHOL.IS.  Strip "-E" immediately before ".IS".
    before_norm = combined["Ticker"].copy()
    combined["Ticker"] = combined["Ticker"].str.replace(r"-E\.IS$", ".IS", regex=True)
    normalized = (combined["Ticker"] != before_norm).sum()
    if normalized:
        print(f"[gc] Normalized {normalized} Turkish .IS tickers (stripped -E suffix)")
    combined = combined.drop_duplicates(subset=["Ticker"], keep="first")

    # Apply known ticker overrides (runtime corrections only — symbol renames,
    # corporate actions etc. that can't be detected at CSV build time)
    combined["Ticker"] = combined["Ticker"].replace(TICKER_OVERRIDES)
    combined = combined.drop_duplicates(subset=["Ticker"], keep="first")

    # Drop ghost/placeholder tickers — lightweight safety net for anything
    # that slipped through the MSCI CSV build step (e.g. stale cache entries
    # from before update_msci_world_classification.py was hardened).
    before = len(combined)
    combined = combined[~combined["Ticker"].apply(is_ghost_ticker)]
    dropped = before - len(combined)
    if dropped:
        print(f"[gc] Dropped {dropped} ghost/placeholder tickers from universe")

    return combined.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
# Earnings data download
# ────────────────────────────────────────────────────────────────
def _fetch_eps_method1(tk) -> List[Dict]:
    """Method 1: tk.earnings_dates — scrapes HTML table. Most complete but rate-limit sensitive."""
    ed = tk.earnings_dates
    if ed is None or ed.empty:
        return []
    rows = []
    for idx, row in ed.iterrows():
        eps_est = _safe_float(row.get("EPS Estimate"))
        eps_rep = _safe_float(row.get("Reported EPS"))
        surprise = _safe_float(row.get("Surprise(%)"))
        rev_est = _safe_float(row.get("Revenue Estimate"))
        rev_rep = _safe_float(row.get("Reported Revenue"))
        rows.append({
            "date": str(pd.Timestamp(idx).date()) if not pd.isna(idx) else None,
            "eps_estimate": eps_est if np.isfinite(eps_est) else None,
            "eps_reported": eps_rep if np.isfinite(eps_rep) else None,
            "eps_surprise_pct": surprise if np.isfinite(surprise) else None,
            "revenue_estimate": rev_est if np.isfinite(rev_est) else None,
            "revenue_reported": rev_rep if np.isfinite(rev_rep) else None,
            "_method": "earnings_dates",
        })
    return rows


def _fetch_eps_method2(tk) -> List[Dict]:
    """Method 2: REMOVED — tk.quarterly_earnings calls tk.earnings internally,
    which is deprecated in yfinance 1.2+ and no longer available via API.
    Kept as a stub so the method_results list indices remain stable.
    Returns [] always.
    """
    return []


def _fetch_eps_method3(tk) -> List[Dict]:
    """Method 3: tk.get_earnings_dates(limit=40) — newer yfinance API, works when method 1 fails."""
    try:
        ed = tk.get_earnings_dates(limit=40)
        if ed is None or ed.empty:
            return []
        rows = []
        for idx, row in ed.iterrows():
            eps_est = _safe_float(row.get("EPS Estimate"))
            eps_rep = _safe_float(row.get("Reported EPS"))
            surprise = _safe_float(row.get("Surprise(%)"))
            rows.append({
                "date": str(pd.Timestamp(idx).date()) if not pd.isna(idx) else None,
                "eps_estimate": eps_est if np.isfinite(eps_est) else None,
                "eps_reported": eps_rep if np.isfinite(eps_rep) else None,
                "eps_surprise_pct": surprise if np.isfinite(surprise) else None,
                "revenue_estimate": None,
                "revenue_reported": None,
                "_method": "get_earnings_dates",
            })
        return rows
    except Exception:
        return []


def _fetch_eps_method4(tk) -> List[Dict]:
    """Method 4: tk.income_stmt (annual) + tk.quarterly_income_stmt — derive EPS from net income / shares.
    Last resort when all other methods fail. No estimate vs actual, but confirms earnings history."""
    try:
        inc = tk.quarterly_income_stmt
        if inc is None or inc.empty:
            return []
        # Try to find net income and shares outstanding
        ni_label = next((l for l in ["Net Income", "NetIncome", "Net Income Common Stockholders"] if l in inc.index), None)
        if ni_label is None:
            return []
        ni_series = inc.loc[ni_label].dropna().sort_index()
        # Get diluted shares if available
        shares_label = next((l for l in ["Diluted Average Shares", "BasicAverageShares", "Ordinary Shares Number"] if l in inc.index), None)
        shares_series = inc.loc[shares_label].dropna().sort_index() if shares_label else None
        rows = []
        for date_col, ni in ni_series.items():
            eps_val = None
            if shares_series is not None and date_col in shares_series.index:
                sh = float(shares_series[date_col])
                if sh and sh > 0:
                    eps_val = round(float(ni) / sh, 4)
            rows.append({
                "date": str(pd.Timestamp(date_col).date()),
                "eps_estimate": None,
                "eps_reported": eps_val,
                "eps_surprise_pct": None,
                "revenue_estimate": None,
                "revenue_reported": None,
                "_method": "income_stmt_derived",
            })
        return rows
    except Exception:
        return []


def _fetch_revenue_estimates_yahoo(tk, ticker: str) -> List[Dict]:
    """Fetch per-quarter revenue estimates + actuals directly from Yahoo Finance
    quoteSummary endpoint, which yfinance 1.x no longer surfaces via earnings_dates.

    Uses two Yahoo modules:
      earningsTrend  → revenueEstimate.avg per quarter (recent + upcoming)
      earningsHistory → epsActual + epsEstimate + sometimes revenueEstimate per past Q

    Returns list of {date, revenue_estimate, revenue_reported, eps_estimate, eps_reported}
    to be merged into earnings_dates rows.

    yfinance 1.x changed from HTML scraping (which showed 5-col table incl. revenue)
    to Yahoo's v1/finance/earnings JSON API (EPS-only). Revenue estimates are still
    available via the quoteSummary v10 endpoint — we just need to call it directly.
    """
    rows = []
    try:
        # Use yfinance's built-in session (handles cookie/crumb auth automatically)
        # Access via tk._data or the shared download session
        session = None
        try:
            session = tk._data.cache  # yfinance 1.x internal
        except Exception:
            pass

        # Build the URL for quoteSummary with relevant modules
        sym = ticker.split(".")[0] if "." not in ticker else ticker
        url = (
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
            f"?modules=earningsTrend%2CearningsHistory&corsDomain=finance.yahoo.com"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://finance.yahoo.com/quote/{sym}/analysis",
        }

        # Try with yfinance session first (has auth), fallback to plain requests
        raw = None
        try:
            import requests as _req
            resp = _req.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                raw = resp.json()
        except Exception:
            pass

        if not raw:
            import urllib.request, urllib.parse
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = json.loads(r.read().decode("utf-8"))

        if not raw:
            return rows

        result = raw.get("quoteSummary", {}).get("result", [])
        if not result:
            return rows
        data = result[0]

        # 1) earningsTrend: current/recent quarters with revenue estimates
        trend_by_qtr: Dict[str, Dict] = {}
        trend = data.get("earningsTrend", {}).get("trend", [])
        for item in trend:
            period = item.get("endDate", {}).get("fmt") or item.get("endDate", "")
            if isinstance(period, dict):
                period = period.get("fmt", "")
            d = str(period)[:7]  # YYYY-MM
            if not d or len(d) < 7:
                continue
            rev_est = None
            rev_avg = (item.get("revenueEstimate") or {})
            if isinstance(rev_avg, dict):
                rev_est = _safe_float(rev_avg.get("avg") or rev_avg.get("raw"))
            eps_est = None
            eps_avg = (item.get("earningsEstimate") or {})
            if isinstance(eps_avg, dict):
                eps_est = _safe_float(eps_avg.get("avg") or eps_avg.get("raw"))
            trend_by_qtr[d] = {
                "rev_estimate": rev_est if (rev_est is not None and np.isfinite(rev_est) and rev_est > 0) else None,
                "eps_estimate": eps_est if (eps_est is not None and np.isfinite(eps_est)) else None,
            }

        # 2) earningsHistory: past quarters with actuals + estimates
        hist_by_qtr: Dict[str, Dict] = {}
        history = data.get("earningsHistory", {}).get("history", [])
        for item in history:
            period = item.get("quarter", {})
            if isinstance(period, dict):
                period = period.get("fmt", "")
            d = str(period)[:7]
            if not d or len(d) < 7:
                continue
            eps_act = _safe_float((item.get("epsActual") or {}).get("raw")
                                  if isinstance(item.get("epsActual"), dict)
                                  else item.get("epsActual"))
            eps_est_h = _safe_float((item.get("epsEstimate") or {}).get("raw")
                                    if isinstance(item.get("epsEstimate"), dict)
                                    else item.get("epsEstimate"))
            # Revenue fields in earningsHistory (present on some tickers)
            rev_act = _safe_float((item.get("revenueActual") or {}).get("raw")
                                  if isinstance(item.get("revenueActual"), dict)
                                  else item.get("revenueActual"))
            rev_est_h = _safe_float((item.get("revenueEstimate") or {}).get("raw")
                                    if isinstance(item.get("revenueEstimate"), dict)
                                    else item.get("revenueEstimate"))
            hist_by_qtr[d] = {
                "eps_actual": eps_act if (eps_act is not None and np.isfinite(eps_act)) else None,
                "eps_estimate": eps_est_h if (eps_est_h is not None and np.isfinite(eps_est_h)) else None,
                "rev_actual": rev_act if (rev_act is not None and np.isfinite(rev_act) and rev_act > 0) else None,
                "rev_estimate": rev_est_h if (rev_est_h is not None and np.isfinite(rev_est_h) and rev_est_h > 0) else None,
            }

        # Merge trend + history into unified rows
        all_qtrs = set(trend_by_qtr.keys()) | set(hist_by_qtr.keys())
        for d in all_qtrs:
            t_data = trend_by_qtr.get(d, {})
            h_data = hist_by_qtr.get(d, {})
            row = {
                "date": d + "-01",  # approximate date for YYYY-MM matching
                "revenue_estimate": t_data.get("rev_estimate") or h_data.get("rev_estimate"),
                "revenue_reported": h_data.get("rev_actual"),
                "eps_estimate":     t_data.get("eps_estimate") or h_data.get("eps_estimate"),
                "eps_reported":     h_data.get("eps_actual"),
                "_method":          "yahoo_quotesummary",
            }
            if row["revenue_estimate"] is not None or row["revenue_reported"] is not None:
                rows.append(row)

    except Exception as _e:
        pass  # silent — caller will fall through to other sources

    return rows


def fetch_catalyst_events(ticker: str, tk) -> List[Dict]:
    """Scan recent yfinance news for major catalyst events (FDA approvals, large contracts,
    geopolitical shocks) that qualify as Layer-2 equivalents without earnings data.

    Returns list of {date, headline, catalyst_tier, catalyst_type, relevance_score}
    Only returns items scored as genuinely market-moving (tier1 match).
    """
    events = []
    try:
        news = tk.news or []
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=180)
        for item in news[:30]:  # Only check last 30 articles
            title = str(item.get("title", "")).lower()
            pub_ts = item.get("providerPublishTime", 0)
            pub_date = dt.datetime.fromtimestamp(pub_ts, tz=dt.timezone.utc) if pub_ts else None
            if pub_date and pub_date < cutoff:
                continue

            tier1_hit = next((kw for kw in CATALYST_KEYWORDS_TIER1 if kw in title), None)
            tier2_hit = next((kw for kw in CATALYST_KEYWORDS_TIER2 if kw in title), None)

            if tier1_hit:
                # Classify the catalyst type
                ctype = "regulatory"
                if any(k in title for k in ["contract", "defense", "government", "nato"]):
                    ctype = "contract"
                elif any(k in title for k in ["launch", "enters market", "commercial"]):
                    ctype = "product_launch"
                elif any(k in title for k in ["acquisition", "merger", "takeover", "buyout"]):
                    ctype = "ma"
                elif any(k in title for k in ["sanctions", "export ban", "opec", "supply shock"]):
                    ctype = "geopolitical"

                events.append({
                    "date": pub_date.date().isoformat() if pub_date else None,
                    "headline": item.get("title", ""),
                    "catalyst_tier": 1,
                    "catalyst_type": ctype,
                    "keyword_matched": tier1_hit,
                    "url": item.get("link", ""),
                })
            elif tier2_hit and not events:
                # Only include tier2 if no tier1 found — softer signal
                events.append({
                    "date": pub_date.date().isoformat() if pub_date else None,
                    "headline": item.get("title", ""),
                    "catalyst_tier": 2,
                    "catalyst_type": "performance",
                    "keyword_matched": tier2_hit,
                    "url": item.get("link", ""),
                })
    except Exception:
        pass
    return events


def _best_eps_rows(methods_results: List[List[Dict]]) -> Tuple[List[Dict], str]:
    """Pick the best EPS data from multiple method results.
    Prefers methods with most past (reported) data points."""
    best: List[Dict] = []
    best_method = "none"
    for rows in methods_results:
        past = [r for r in rows if r.get("eps_reported") is not None]
        if len(past) > len([r for r in best if r.get("eps_reported") is not None]):
            best = rows
            best_method = rows[0].get("_method", "unknown") if rows else "unknown"
    return best, best_method


def fetch_earnings_data(ticker: str) -> Dict[str, Any]:
    """Fetch earnings and revenue data for a single ticker from yfinance.
    Uses 4 fallback methods for EPS + catalyst news scan.

    Returns dict with:
        - quarterly_revenue: list of {date, revenue, revenue_yoy_growth}
        - earnings_dates: list of {date, eps_estimate, eps_reported, eps_surprise_pct}
        - catalyst_events: list of {date, headline, catalyst_tier, catalyst_type}
        - eps_method: which yfinance method yielded the best EPS data
        - info: {revenue_growth, earnings_growth, ...} from Ticker.info
        - error: str if failed
    """
    out: Dict[str, Any] = {"ticker": ticker}
    try:
        tk = yf.Ticker(ticker)

        # 1) Quarterly income statement → revenue
        try:
            inc = tk.quarterly_income_stmt
            if inc is not None and not inc.empty:
                rev_rows = []
                rev_label = None
                for label in ["Total Revenue", "Revenue", "TotalRevenue"]:
                    if label in inc.index:
                        rev_label = label
                        break
                if rev_label is not None:
                    rev_series = inc.loc[rev_label].dropna().sort_index()
                    for date_col, val in rev_series.items():
                        rev_rows.append({
                            "date": str(pd.Timestamp(date_col).date()),
                            "revenue": _safe_float(val),
                        })
                    # Compute YoY growth (compare to same quarter 1 year ago)
                    for i, row in enumerate(rev_rows):
                        row["revenue_yoy_growth"] = None
                        if i >= 4:
                            prev_rev = rev_rows[i - 4]["revenue"]
                            if prev_rev and prev_rev > 0 and np.isfinite(prev_rev):
                                curr_rev = row["revenue"]
                                if np.isfinite(curr_rev):
                                    row["revenue_yoy_growth"] = round(
                                        (curr_rev / prev_rev - 1.0) * 100.0, 2
                                    )
                out["quarterly_revenue"] = rev_rows
            else:
                out["quarterly_revenue"] = []
        except Exception as e:
            out["quarterly_revenue"] = []
            out["_rev_error"] = str(e)

        # 1b) Annual revenue fallback — for markets that report semi-annually
        # or where quarterly_income_stmt returns empty (Japan .T, South Africa .JO,
        # parts of .HK, .AX). Divide annual revenue by 4 as proxy for quarterly.
        # Flagged with revenue_source="annual_estimated" so consumers know.
        if not out.get("quarterly_revenue"):
            _ANNUAL_FALLBACK_SUFFIXES = {"T", "JO", "HK", "AX", "L", "PA", "SW"}
            t_suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else "US"
            if t_suffix in _ANNUAL_FALLBACK_SUFFIXES:
                try:
                    ann = tk.income_stmt
                    if ann is not None and not ann.empty:
                        rev_label = None
                        for label in ["Total Revenue", "Revenue", "TotalRevenue"]:
                            if label in ann.index:
                                rev_label = label
                                break
                        if rev_label is not None:
                            ann_series = ann.loc[rev_label].dropna().sort_index()
                            ann_rows = []
                            for date_col, val in ann_series.items():
                                ann_rows.append({
                                    "date": str(pd.Timestamp(date_col).date()),
                                    "revenue": _safe_float(val / 4.0),  # annualised ÷ 4
                                    "revenue_yoy_growth": None,
                                    "revenue_source": "annual_estimated",
                                })
                            # Compute YoY on annual estimates
                            for i, row in enumerate(ann_rows):
                                if i >= 1:
                                    prev = ann_rows[i - 1]["revenue"]
                                    curr = row["revenue"]
                                    if prev and prev > 0 and np.isfinite(prev) and np.isfinite(curr):
                                        row["revenue_yoy_growth"] = round(
                                            (curr / prev - 1.0) * 100.0, 2
                                        )
                            if ann_rows:
                                out["quarterly_revenue"] = ann_rows
                                out["_rev_fallback"] = "annual_estimated"
                except Exception:
                    pass

        # 2) EPS beat/miss — try 4 methods in order, pick best result
        method_results = []
        errors = []

        try:
            r1 = _fetch_eps_method1(tk)
            method_results.append(r1)
        except Exception as e:
            errors.append(f"m1:{e}")
            method_results.append([])

        # Only try method 2 if method 1 returned no past data
        past1 = [r for r in method_results[0] if r.get("eps_reported") is not None]
        if len(past1) < 2:
            try:
                time.sleep(0.1)
                r2 = _fetch_eps_method2(tk)
                method_results.append(r2)
            except Exception as e:
                errors.append(f"m2:{e}")
                method_results.append([])
        else:
            method_results.append([])

        # Method 3 only if still weak
        past_so_far = max(
            len([r for r in method_results[0] if r.get("eps_reported") is not None]),
            len([r for r in method_results[1] if r.get("eps_reported") is not None]),
        )
        if past_so_far < 2:
            try:
                time.sleep(0.15)
                r3 = _fetch_eps_method3(tk)
                method_results.append(r3)
            except Exception as e:
                errors.append(f"m3:{e}")
                method_results.append([])
        else:
            method_results.append([])

        # Method 4 (derived from income stmt) only as absolute last resort
        all_past = max(
            len([r for r in m if r.get("eps_reported") is not None])
            for m in method_results
        )
        if all_past < 1:
            try:
                r4 = _fetch_eps_method4(tk)
                method_results.append(r4)
            except Exception as e:
                errors.append(f"m4:{e}")
                method_results.append([])

        best_eps, best_method = _best_eps_rows(method_results)

        # ── Method 1b: Yahoo quoteSummary revenue enrichment ─────────────
        # yfinance 1.x earnings_dates uses v1/finance/earnings — EPS-only JSON.
        # Revenue estimates lived in the old 0.2.x HTML table but were dropped
        # when yfinance switched to Yahoo's API backend in 1.0.
        # We call quoteSummary earningsTrend + earningsHistory directly to
        # recover revenue_estimate and revenue_reported per past quarter.
        # This is a separate call so it doesn't interfere with EPS method selection.
        try:
            yq_rows = _fetch_revenue_estimates_yahoo(tk, ticker)
            if yq_rows:
                yq_by_qtr: Dict[str, Dict] = {}
                for r in yq_rows:
                    d = (r.get("date") or "")[:7]
                    if d:
                        yq_by_qtr[d] = r
                for row in best_eps:
                    d = (row.get("date") or "")[:7]
                    candidates = [d]
                    try:
                        y, m = int(d[:4]), int(d[5:7])
                        for delta in [-1, -2, 1, 2]:
                            nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                            candidates.append(f"{ny:04d}-{nm:02d}")
                    except Exception:
                        pass
                    for key in candidates:
                        if key in yq_by_qtr:
                            qd = yq_by_qtr[key]
                            if row.get("revenue_estimate") is None and qd.get("revenue_estimate") is not None:
                                row["revenue_estimate"] = qd["revenue_estimate"]
                                row["_rev_est_source"] = "yahoo_qs"
                            if row.get("revenue_reported") is None and qd.get("revenue_reported") is not None:
                                row["revenue_reported"] = qd["revenue_reported"]
                                row["_rev_act_source"] = "yahoo_qs"
                            if row.get("eps_estimate") is None and qd.get("eps_estimate") is not None:
                                row["eps_estimate"] = qd["eps_estimate"]
                                row["_eps_est_source"] = "yahoo_qs"
                            break
                out["_yahoo_qs_rev_rows"] = len([r for r in yq_rows if r.get("revenue_estimate") is not None])
        except Exception as _yq_err:
            out["_yahoo_qs_error"] = str(_yq_err)
        out["earnings_dates"] = best_eps
        out["eps_method"] = best_method
        if errors:
            out["_eps_errors"] = errors

        # 3) Catalyst events scan (news-based, for Layer-2 non-earnings triggers)
        try:
            out["catalyst_events"] = fetch_catalyst_events(ticker, tk)
        except Exception as e:
            out["catalyst_events"] = []
            out["_catalyst_error"] = str(e)

        # 4) Info snapshot → current growth rates
        try:
            info = tk.info or {}
            out["info"] = {
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "sector": str(info.get("sector", "")),
                "industry": str(info.get("industry", "")),
                "market_cap": _safe_float(info.get("marketCap")),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "short_name": str(info.get("shortName", "")),
                "currency": str(info.get("currency", "")),
            }
        except Exception as e:
            out["info"] = {}
            out["_info_error"] = str(e)

    except Exception as e:
        out["error"] = str(e)

    # ── EPS + Revenue estimate enrichment cascade ────────────────────────────
    # Both EPS AND revenue estimates can be missing from yfinance:
    #   - revenue_estimate: missing for ~88% of universe since yfinance 1.2.0
    #   - eps_estimate: missing when method 4 (income_stmt_derived) runs,
    #     or for non-US tickers where yfinance returns eps_reported only
    # We enrich both fields symmetrically across all fallback sources.
    # Source tagging: _eps_est_source / _rev_est_source / _rev_act_source on each row.

    for r in out.get("earnings_dates", []):
        if r.get("eps_estimate") is not None:
            r.setdefault("_eps_est_source", "yfinance")
        if r.get("revenue_estimate") is not None:
            r.setdefault("_rev_est_source", "yfinance")
        if r.get("revenue_reported") is not None:
            r.setdefault("_rev_act_source", "yfinance")

    def _missing_estimates_count():
        return sum(1 for r in out.get("earnings_dates", [])
                   if r.get("eps_reported") is not None
                   and (r.get("eps_estimate") is None
                        or r.get("revenue_estimate") is None
                        or r.get("revenue_reported") is None))

    # 5a) investing.com — no API key. Fills EPS + revenue together.
    if _missing_estimates_count() > 0:
        try:
            filled_ic = enrich_estimates_investing_com(
                out["earnings_dates"], ticker,
                quarterly_revenue=out.get("quarterly_revenue"),
            )
            if filled_ic:
                out["_investing_com_filled"] = filled_ic
        except Exception as _ic_err:
            out["_investing_com_error"] = str(_ic_err)

    # 5b) FMP analyst-estimates — fills both EPS (estimatedEpsAvg) and revenue (estimatedRevenueAvg).
    # Two URL patterns tried: 'stable' endpoint first (newer), then 'api/v3' (classic).
    # Free tier covers US tickers for analyst-estimates. Non-US coverage varies.
    fmp_key_enrich = os.environ.get("FMP_API_KEY", "").strip()
    if fmp_key_enrich and _missing_estimates_count() > 0 and out.get("earnings_dates"):
        try:
            from urllib.parse import urlencode
            import urllib.request as _ureq
            sym_fmp = out.get("fmp_symbol") or ticker.split(".")[0].upper()
            est_data = None
            qs = urlencode({"symbol": sym_fmp, "period": "quarter", "limit": 16,
                            "apikey": fmp_key_enrich})
            # Try stable endpoint first, fall back to api/v3
            for base_url in [f"{_FMP_BASE}/analyst-estimates",
                              f"https://financialmodelingprep.com/api/v3/analyst-estimates"]:
                try:
                    with _ureq.urlopen(f"{base_url}?{qs}", timeout=8) as r:
                        raw = json.loads(r.read().decode())
                        if isinstance(raw, list) and raw:
                            est_data = raw
                            break
                except Exception:
                    continue
            if est_data:
                fmp_est_by_qtr = {}
                for q in est_data:
                    d = str(q.get("date", ""))[:7]
                    eps_avg = _safe_float(q.get("estimatedEpsAvg") or q.get("epsAvg"))
                    rev_avg = _safe_float(q.get("estimatedRevenueAvg") or q.get("revenueAvg"))
                    if d:
                        fmp_est_by_qtr[d] = {"eps": eps_avg, "rev": rev_avg}
                rev_act_by_qtr = {(qr.get("date", ""))[:7]: qr["revenue"]
                                  for qr in out.get("quarterly_revenue", [])
                                  if qr.get("revenue") is not None}
                filled_fmp = 0
                for row in out["earnings_dates"]:
                    d = (row.get("date") or "")[:7]
                    candidates = [d]
                    try:
                        y, m = int(d[:4]), int(d[5:7])
                        for delta in [-1, -2, 1, 2]:
                            nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                            candidates.append(f"{ny:04d}-{nm:02d}")
                    except Exception: pass
                    for key in candidates:
                        if key not in fmp_est_by_qtr:
                            continue
                        q_est = fmp_est_by_qtr[key]
                        if (row.get("eps_estimate") is None
                                and q_est["eps"] is not None and np.isfinite(q_est["eps"])):
                            row["eps_estimate"] = q_est["eps"]
                            row["_eps_est_source"] = "fmp"
                            filled_fmp += 1
                            est = _safe_float(row["eps_estimate"])
                            rep = _safe_float(row.get("eps_reported"))
                            if np.isfinite(est) and np.isfinite(rep) and abs(est) > 0.001:
                                row["eps_surprise_pct"] = round((rep / est - 1.0) * 100.0, 2)
                        if (row.get("revenue_estimate") is None
                                and q_est["rev"] is not None
                                and np.isfinite(q_est["rev"]) and q_est["rev"] > 0):
                            row["revenue_estimate"] = q_est["rev"]
                            row["_rev_est_source"] = "fmp"
                            filled_fmp += 1
                        if row.get("revenue_reported") is None and key in rev_act_by_qtr:
                            row["revenue_reported"] = rev_act_by_qtr[key]
                            row["_rev_act_source"] = "fmp"
                        break
                if filled_fmp:
                    out["_fmp_rev_estimates_filled"] = filled_fmp  # fixed key name (was _fmp_enrich_filled)
        except Exception as _fmp_enr_err:
            out["_fmp_enrich_error"] = str(_fmp_enr_err)

    # 5c) Finnhub — final fallback. Enriches both EPS and revenue.
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if finnhub_key and _missing_estimates_count() > 0:
        try:
            filled_fh = enrich_estimates_finnhub(out["earnings_dates"], ticker, finnhub_key)
            if filled_fh:
                out["_finnhub_filled"] = filled_fh
        except Exception as _fh_err:
            out["_finnhub_error"] = str(_fh_err)

    out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # Clear new_constituent flag — ticker has now been fetched at least once
    out.pop("new_constituent", None)
    return out
    """Fetch earnings and revenue data for a single ticker from yfinance.

    Returns dict with:
        - quarterly_revenue: list of {date, revenue, revenue_yoy_growth}
        - earnings_dates: list of {date, eps_estimate, eps_reported, eps_surprise_pct}
        - info: {revenue_growth, earnings_growth} from Ticker.info
        - error: str if failed
    """
    out: Dict[str, Any] = {"ticker": ticker}
    try:
        tk = yf.Ticker(ticker)

        # 1) Quarterly income statement → revenue
        try:
            inc = tk.quarterly_income_stmt
            if inc is not None and not inc.empty:
                rev_rows = []
                # inc columns are dates (most recent first), rows are line items
                rev_label = None
                for label in ["Total Revenue", "Revenue", "TotalRevenue"]:
                    if label in inc.index:
                        rev_label = label
                        break
                if rev_label is not None:
                    rev_series = inc.loc[rev_label].dropna()
                    # Sort chronologically (oldest first)
                    rev_series = rev_series.sort_index()
                    for date_col, val in rev_series.items():
                        rev_rows.append({
                            "date": str(pd.Timestamp(date_col).date()),
                            "revenue": _safe_float(val),
                        })
                    # Compute YoY growth (compare to same quarter 1 year ago)
                    for i, row in enumerate(rev_rows):
                        row["revenue_yoy_growth"] = None
                        # Find the quarter ~4 quarters back (same quarter prev year)
                        if i >= 4:
                            prev_rev = rev_rows[i - 4]["revenue"]
                            if prev_rev and prev_rev > 0 and np.isfinite(prev_rev):
                                curr_rev = row["revenue"]
                                if np.isfinite(curr_rev):
                                    row["revenue_yoy_growth"] = round(
                                        (curr_rev / prev_rev - 1.0) * 100.0, 2
                                    )
                out["quarterly_revenue"] = rev_rows
            else:
                out["quarterly_revenue"] = []
        except Exception as e:
            out["quarterly_revenue"] = []
            out["_rev_error"] = str(e)

        # 2) Earnings dates → EPS beat/miss
        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                eps_rows = []
                for idx, row in ed.iterrows():
                    eps_est = _safe_float(row.get("EPS Estimate"))
                    eps_rep = _safe_float(row.get("Reported EPS"))
                    surprise = _safe_float(row.get("Surprise(%)"))
                    # Also check for revenue estimate/actual if available
                    rev_est = _safe_float(row.get("Revenue Estimate"))
                    rev_rep = _safe_float(row.get("Reported Revenue"))
                    eps_rows.append({
                        "date": str(pd.Timestamp(idx).date()) if not pd.isna(idx) else None,
                        "eps_estimate": eps_est if np.isfinite(eps_est) else None,
                        "eps_reported": eps_rep if np.isfinite(eps_rep) else None,
                        "eps_surprise_pct": surprise if np.isfinite(surprise) else None,
                        "revenue_estimate": rev_est if np.isfinite(rev_est) else None,
                        "revenue_reported": rev_rep if np.isfinite(rev_rep) else None,
                    })
                out["earnings_dates"] = eps_rows
            else:
                out["earnings_dates"] = []
        except Exception as e:
            out["earnings_dates"] = []
            out["_ed_error"] = str(e)

        # 3) Info snapshot → current growth rates
        try:
            info = tk.info or {}
            out["info"] = {
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "sector": str(info.get("sector", "")),
                "industry": str(info.get("industry", "")),
                "market_cap": _safe_float(info.get("marketCap")),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "short_name": str(info.get("shortName", "")),
            }
        except Exception as e:
            out["info"] = {}
            out["_info_error"] = str(e)

    except Exception as e:
        out["error"] = str(e)

    out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out


# ────────────────────────────────────────────────────────────────
# FMP (Financial Modeling Prep) fallback data layer
# Activated when FMP_API_KEY env var is set.
# Free tier: 250 API calls/day — plenty for UK/Korea/Japan fallback.
# Starter tier (~$15/mo): 300 calls/min — enough for full universe.
#
# FMP ticker format vs Yahoo Finance:
#   Yahoo .L  → FMP  (no suffix, LSE symbols)   e.g. AZN.L  → AZN
#   Yahoo .T  → FMP  .T                          e.g. 6758.T → 6758.T  (same)
#   Yahoo .KS → FMP  .KS                         e.g. 000660.KS (same)
#   Yahoo .HK → FMP  .HK                         e.g. 0700.HK (same)
#   Yahoo .PA → FMP  .PA                         e.g. MC.PA  (same)
#   Yahoo .DE → FMP  .DE                         e.g. SAP.DE (same)
# ────────────────────────────────────────────────────────────────

# Exchange suffix remapping: Yahoo suffix → FMP suffix (or None = strip suffix)
_FMP_SUFFIX_MAP: Dict[str, Optional[str]] = {
    "L":  None,   # LSE: strip .L for FMP (AZN.L → AZN)
    "KS": "KS",   # Korea: keep .KS
    "T":  "T",    # Japan TSE: keep .T
    "HK": "HK",   # Hong Kong: keep .HK
    "TW": "TW",   # Taiwan: keep .TW
    "PA": "PA",   # Paris: keep .PA
    "DE": "DE",   # Frankfurt XETRA: keep .DE
    "SW": "SW",   # Switzerland: keep .SW
    "AS": "AS",   # Amsterdam: keep .AS
    "MI": "MI",   # Milan: keep .MI
    "MC": "MC",   # Madrid: keep .MC
    "ST": "ST",   # Stockholm: keep .ST
    "OL": "OL",   # Oslo: keep .OL
    "HE": "HE",   # Helsinki: keep .HE
    "CO": "CO",   # Copenhagen: keep .CO
    "TO": "TO",   # Toronto: keep .TO
    "AX": "AX",   # Australia ASX: keep .AX
    "NS": "NS",   # India NSE: keep .NS
    "SA": "SA",   # Brazil B3: keep .SA
}

_FMP_BASE = "https://financialmodelingprep.com/stable"


def _yahoo_to_fmp(ticker: str) -> str:
    """Convert Yahoo Finance ticker to FMP ticker format."""
    if "." not in ticker:
        return ticker  # US ticker — same in both
    base, suffix = ticker.rsplit(".", 1)
    fmp_suffix = _FMP_SUFFIX_MAP.get(suffix)
    if fmp_suffix is None and suffix in _FMP_SUFFIX_MAP:
        return base           # Explicitly mapped to None = strip (e.g. LSE .L)
    elif suffix not in _FMP_SUFFIX_MAP:
        return f"{base}.{suffix}"   # Unknown suffix — pass through as-is
    return f"{base}.{fmp_suffix}"


def _fmp_get(path: str, params: Dict, api_key: str) -> Any:
    """Single FMP API call with basic error handling. Returns parsed JSON or None."""
    import urllib.request
    import urllib.parse
    params = {**params, "apikey": api_key}
    qs = urllib.parse.urlencode(params)
    url = f"{_FMP_BASE}{path}?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("Error Message"):
                return None  # Invalid key or symbol
            return data
    except Exception:
        return None


_FINNHUB_BASE = "https://finnhub.io/api/v1"

def _finnhub_get(path: str, params: Dict, api_key: str) -> Any:
    """Single Finnhub API call. Free tier: 60 calls/min. Returns parsed JSON or None."""
    import urllib.request, urllib.parse
    params = {**params, "token": api_key}
    url = f"{_FINNHUB_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def enrich_estimates_finnhub(earnings_dates: List[Dict], ticker: str, api_key: str) -> int:
    """Fill EPS + revenue estimates and actuals from Finnhub /stock/earnings.

    Finnhub returns per-quarter: epsEstimate, epsActual, revenueEstimate, revenueActual.
    Enriches BOTH EPS and revenue fields symmetrically — not just revenue.
    Free tier: US stocks well covered. International coverage varies.
    Returns count of individual fields filled.
    """
    if not api_key or not earnings_dates:
        return 0
    # Only call if something is still missing (EPS or revenue)
    missing = [r for r in earnings_dates
               if r.get("eps_reported") is not None
               and (r.get("eps_estimate") is None
                    or r.get("revenue_estimate") is None
                    or r.get("revenue_reported") is None)]
    if not missing:
        return 0
    sym = ticker.split(".")[0]  # Finnhub uses bare symbol for most US tickers
    data = _finnhub_get("/stock/earnings", {"symbol": sym, "limit": 16}, api_key)
    if not data or not isinstance(data, list):
        return 0
    # Build YYYY-MM -> {eps_est, eps_act, rev_est, rev_act} from Finnhub
    fh_by_qtr = {}
    for q in data:
        d = str(q.get("period") or q.get("date") or "")[:7]
        if d:
            fh_by_qtr[d] = {
                "eps_est": _safe_float(q.get("epsEstimate") or q.get("estimatedEps")),
                "eps_act": _safe_float(q.get("epsActual") or q.get("actualEps")),
                "rev_est": _safe_float(q.get("revenueEstimate") or q.get("estimatedRevenue")),
                "rev_act": _safe_float(q.get("revenueActual") or q.get("actualRevenue")),
            }
    filled = 0
    for row in earnings_dates:
        d = (row.get("date") or "")[:7]
        candidates = [d]
        try:
            y, m = int(d[:4]), int(d[5:7])
            for delta in [-1, -2, 1, 2]:
                nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                candidates.append(f"{ny:04d}-{nm:02d}")
        except Exception: pass
        for key in candidates:
            if key in fh_by_qtr:
                q_data = fh_by_qtr[key]
                # EPS fields
                if (row.get("eps_estimate") is None
                        and q_data["eps_est"] is not None
                        and np.isfinite(q_data["eps_est"])):
                    row["eps_estimate"] = q_data["eps_est"]
                    row["_eps_est_source"] = "finnhub"
                    filled += 1
                    # Recompute surprise
                    est = _safe_float(row["eps_estimate"])
                    rep = _safe_float(row.get("eps_reported"))
                    if np.isfinite(est) and np.isfinite(rep) and abs(est) > 0.001:
                        row["eps_surprise_pct"] = round((rep / est - 1.0) * 100.0, 2)
                # Revenue fields
                if (row.get("revenue_estimate") is None
                        and q_data["rev_est"] is not None
                        and np.isfinite(q_data["rev_est"])):
                    row["revenue_estimate"] = q_data["rev_est"]
                    row["_rev_est_source"] = "finnhub"
                    filled += 1
                if (row.get("revenue_reported") is None
                        and q_data["rev_act"] is not None
                        and np.isfinite(q_data["rev_act"])):
                    row["revenue_reported"] = q_data["rev_act"]
                    row["_rev_act_source"] = "finnhub"
                break
    return filled


def enrich_estimates_investing_com(
    earnings_dates: List[Dict],
    ticker: str,
    quarterly_revenue: Optional[List[Dict]] = None,
) -> int:
    """Scrape EPS + revenue estimates and actuals from investing.com earnings page.

    Uses curl_cffi (Chrome TLS impersonation) to bypass Cloudflare bot detection.
    curl_cffi is already installed as a yfinance 1.2 dependency — no extra packages needed.

    Two-step approach:
      1. Search API: resolve ticker symbol → correct investing.com slug + pairId
         GET https://api.investing.com/api/search/v2/search?q={symbol}&type=quotes&limit=6
      2. Earnings page: GET https://www.investing.com/equities/{slug}-earnings
         Parse HTML table: Date | EPS Est | EPS Act | Rev Est | Rev Act | Surprise%
      3. Fallback: investing.com internal earnings-history API using pairId

    Why curl_cffi: regular requests/urllib gets Cloudflare 403 on GitHub Actions IPs.
    curl_cffi mimics Chrome's TLS fingerprint (JA3/JA4) which passes Cloudflare's
    bot detection. This is the same approach yfinance uses internally for auth.

    Args:
        earnings_dates:    List of per-quarter dicts to enrich with EPS/rev estimates.
                           Rows without eps_reported are skipped.
        ticker:            Yahoo Finance ticker (e.g. 'NVDA', 'AZN.L').
        quarterly_revenue: If provided (mutable list), revenue actuals from investing.com
                           are APPENDED for quarters not already present. This extends
                           yfinance's hard 5-quarter cap — investing.com covers ~16-20 quarters.
                           Caller should pass the existing list; this function extends it.

    Returns:
        Count of fields filled across all earnings_dates rows.
        quarterly_revenue is mutated in-place (appended) when provided.
    """
    missing = [r for r in earnings_dates
               if r.get("eps_reported") is not None
               and (r.get("eps_estimate") is None
                    or r.get("revenue_estimate") is None
                    or r.get("revenue_reported") is None)]
    # Also run if quarterly_revenue is shallow (< 8 quarters) — IC extends history
    qr_shallow = (quarterly_revenue is not None
                  and len([r for r in quarterly_revenue if r.get("revenue") is not None]) < 8)
    if not missing and not qr_shallow:
        return 0

    # ── 0. Import curl_cffi — installed as yfinance 1.x dependency ──────────
    try:
        from curl_cffi import requests as cffi_req
    except ImportError:
        # Graceful fallback: curl_cffi not available — skip silently
        return 0

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return 0

    bare = ticker.split(".")[0].upper()   # NVDA, AAPL etc.
    impersonate = "chrome124"

    # ── 1. Slug discovery ────────────────────────────────────────────────
    # Strategy: try common slug patterns directly rather than relying on
    # the search API (which changed response format and now returns articles).
    # investing.com slugs follow predictable patterns:
    #   NVDA  → nvidia-corp
    #   AAPL  → apple-computer-inc
    #   MSFT  → microsoft-corp
    # We try the bare ticker lowercased, then known overrides, then skip slug step.
    slug = None
    pair_id = None

    # Known slug overrides for common tickers where the slug is non-obvious
    SLUG_OVERRIDES: Dict[str, str] = {
        # ── Mega-cap tech ──────────────────────────────────────────────────
        "AAPL":  "apple-computer-inc",
        "MSFT":  "microsoft-corp",
        "NVDA":  "nvidia-corp",
        "GOOGL": "alphabet-inc-cl-a",
        "GOOG":  "alphabet-inc-cl-c",
        "META":  "facebook-inc",
        "AMZN":  "amazon-com-inc",
        "TSLA":  "tesla-motors",
        "AVGO":  "broadcom-ltd",
        # ── Nasdaq 100 — semiconductors ────────────────────────────────────
        "AMD":   "advanced-micro-devices",
        "QCOM":  "qualcomm-inc",
        "AMAT":  "applied-materials",
        "MU":    "micron-technology",
        "KLAC":  "kla-tencor-corp",
        "LRCX":  "lam-research",
        "MRVL":  "marvell-technology",
        "NXPI":  "nxp-semiconductors",
        "MCHP":  "microchip-technology",
        "ADI":   "analog-devices",
        "ON":    "on-semiconductor",
        "GFS":   "globalfoundries",
        "SMCI":  "super-micro-computer",
        # ── Software / cloud ───────────────────────────────────────────────
        "INTU":  "intuit-inc",
        "CSCO":  "cisco-sys-inc",
        "PANW":  "palo-alto-networks",
        "SNPS":  "synopsys-inc",
        "CDNS":  "cadence-design-systems",
        "WDAY":  "workday-inc",
        "CRWD":  "crowdstrike-holdings",
        "DDOG":  "datadog-inc",
        "FTNT":  "fortinet-inc",
        "ZS":    "zscaler",
        "TEAM":  "atlassian",
        "ANSS":  "ansys-inc",
        "MDB":   "mongodb",
        "PYPL":  "paypal-holdings",
        "TTD":   "the-trade-desk",
        "ABNB":  "airbnb",
        "DASH":  "doordash",
        "COIN":  "coinbase-global",
        "PLTR":  "palantir-technologies",
        "APP":   "applovin",
        "RBLX":  "roblox",
        "ZM":    "zoom-video-communications",
        "MTCH":  "match-group",
        # ── Consumer / retail ──────────────────────────────────────────────
        "COST":  "costco-wholesale",
        "NFLX":  "netflix-inc",
        "SBUX":  "starbucks-corp",
        "ORLY":  "oreilly-automotive",
        "ROST":  "ross-stores-inc",
        "DLTR":  "dollar-tree-inc",
        "KHC":   "kraft-heinz",
        "MNST":  "monster-beverage",
        "KDP":   "keurig-dr-pepper",
        # ── Healthcare / biotech ───────────────────────────────────────────
        "AMGN":  "amgen-inc",
        "GILD":  "gilead-sciences",
        "VRTX":  "vertex-pharmaceuticals",
        "REGN":  "regeneron-pharmaceuticals",
        "ISRG":  "intuitive-surgical-inc",
        "IDXX":  "idexx-laboratories",
        "BIIB":  "biogen-idec-inc",
        "ILMN":  "illumina-inc",
        "MDLZ":  "mondelez-international",
        "DXCM":  "dexcom",
        "ALGN":  "align-technology",
        "GEHC":  "ge-healthcare",
        # ── Industrials / other ────────────────────────────────────────────
        "HON":   "honeywell-intl",
        "ADP":   "automatic-data-processing",
        "CTAS":  "cintas-corp",
        "PAYX":  "paychex-inc",
        "FAST":  "fastenal-co",
        "PCAR":  "paccar-inc",
        "CPRT":  "copart-inc",
        "ODFL":  "old-dominion-freight",
        "VRSK":  "verisk-analytics",
        "BKNG":  "priceline-com-inc",
        "MAR":   "marriott-intl",
        "CMCSA": "comcast-corp-new",
        "CHTR":  "charter-communications",
        "WBD":   "warner-bros-discovery",
        "SIRI":  "sirius-xm-holdings",
        "TTWO":  "take-two-interactive",
        # ── Energy / materials ─────────────────────────────────────────────
        "LIN":   "linde-plc",
        "CEG":   "constellation-energy",
        "XEL":   "xcel-energy",
        "EXC":   "exelon-corp",
        "ENPH":  "enphase-energy",
        "FANG":  "diamondback-energy",
        # ── Foreign / ADR ──────────────────────────────────────────────────
        "ASML":  "asml-holding",
        "AZN":   "astrazeneca",
        "PEP":   "pepsico-inc",
        "TXN":   "texas-instruments",
        "MELI":  "mercadolibre",
        "PDD":   "pinduoduo",
        "NTES":  "netease",
        "ROP":   "roper-technologies",
        "CTSH":  "cognizant-technology-solutions",
        "AXON":  "axon-enterprise",
        "CEG":   "constellation-energy",
        # ── EV / clean energy ──────────────────────────────────────────────
        "RIVN":  "rivian-automotive",
        "LCID":  "lucid-group",
    }

    if bare in SLUG_OVERRIDES:
        slug = SLUG_OVERRIDES[bare]
    else:
        # Try search API as best-effort (may return articles now)
        try:
            search_url = (
                f"https://api.investing.com/api/search/v2/search"
                f"?q={bare}&type=quotes&lang=56&limit=6"
            )
            sr = cffi_req.get(
                search_url,
                impersonate=impersonate,
                headers={
                    "Accept": "application/json",
                    "Referer": "https://www.investing.com/",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=10,
            )
            if sr.status_code == 200:
                sdata = sr.json()
                # Search response may have quotes, hits, or data.quotes
                quotes = (sdata.get("quotes") or sdata.get("hits")
                          or sdata.get("data", {}).get("quotes", []) or [])
                for q in quotes:
                    q_sym = (q.get("symbol") or q.get("ticker") or "").upper()
                    q_url = q.get("url") or q.get("link") or ""
                    if q_sym == bare or bare in q_url.upper():
                        parts = q_url.strip("/").split("/")
                        if len(parts) >= 2 and parts[0] in ("equities", "stocks"):
                            slug = parts[1]
                        pair_id = q.get("pairId") or q.get("id") or q.get("pair_id")
                        break
        except Exception:
            pass

    # ── 2. Fetch earnings page HTML with Chrome impersonation ─────────────
    ic_by_qtr: Dict[str, Dict] = {}
    soup = None

    import random

    # Build slug candidates: known override first, then bare ticker lowercase
    slug_candidates = []
    if slug:
        slug_candidates.append(slug)
    if bare.lower() not in slug_candidates:
        slug_candidates.append(bare.lower())

    for s in slug_candidates:
        url = f"https://www.investing.com/equities/{s}-earnings"
        try:
            resp = cffi_req.get(
                url,
                impersonate=impersonate,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.investing.com/equities/",
                    "Cache-Control": "no-cache",
                },
                timeout=25,
                allow_redirects=True,
            )
            # 429 = rate limited — back off 10s and retry once
            if resp.status_code == 429:
                time.sleep(10.0 + random.uniform(0, 4))
                resp = cffi_req.get(
                    url, impersonate=impersonate,
                    headers={"Accept": "text/html",
                             "Referer": "https://www.investing.com/"},
                    timeout=25, allow_redirects=True,
                )
            if resp.status_code == 200 and len(resp.text) > 5000:
                html = resp.text
                # 'Period End' is the most reliable table marker (confirmed from live page)
                if "Period End" in html or "EPS Forecast" in html or "earningsCalendar" in html:
                    soup = BeautifulSoup(html, "lxml")
                    break
            elif resp.status_code in (403, 429):
                break   # Hard block — don't try more slugs
        except Exception:
            continue

    # ── 3. Fallback: investing.com internal earnings-history API ─────────
    if soup is None and pair_id:
        try:
            api_url = "https://api.investing.com/api/financials/historical/earnings-history"
            api_resp = cffi_req.post(
                api_url,
                json={"pairId": pair_id, "limit": 40},
                impersonate=impersonate,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Referer": f"https://www.investing.com/equities/{slug or bare.lower()}-earnings",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=15,
            )
            if api_resp.status_code == 200:
                api_data = api_resp.json()
                items = (api_data.get("data") or api_data.get("earnings")
                         or api_data.get("history") or [])
                for item in items:
                    # Investing.com API fields (vary by version)
                    date_raw = (item.get("releaseDate") or item.get("date")
                                or item.get("period") or "")
                    try:
                        import dateutil.parser as _dp
                        d = _dp.parse(str(date_raw)).strftime("%Y-%m")
                    except Exception:
                        continue
                    def _sf(v):
                        try: return float(str(v).replace(",", "").replace("B","").replace("M","").replace("K",""))
                        except: return None
                    def _scale(v, raw_str):
                        n = _sf(v)
                        if n is None: return None
                        s = str(raw_str).upper()
                        if "B" in s: return n * 1e9
                        if "M" in s: return n * 1e6
                        if "K" in s: return n * 1e3
                        return n
                    rev_est_raw = item.get("revenueEstimate") or item.get("revEstimate")
                    rev_act_raw = item.get("revenueActual")   or item.get("revActual")
                    ic_by_qtr[d] = {
                        "eps_estimate": _sf(item.get("epsEstimate") or item.get("eps_estimate")),
                        "eps_actual":   _sf(item.get("epsActual")   or item.get("eps_actual")),
                        "rev_estimate": _scale(rev_est_raw, rev_est_raw),
                        "rev_actual":   _scale(rev_act_raw, rev_act_raw),
                    }
        except Exception:
            pass

    # ── 4. Parse HTML table if we got a page ─────────────────────────────
    # investing.com earnings page structure (confirmed Mar 2026):
    #   Table headers: ['Release Date', 'Period End', 'EPS', '/Forecast',
    #                   'Revenue', '/Forecast', 'EPS Surprise %', 'Revenue Surprise %']
    #   Col 0 = Release Date  ("Feb 25, 2026")
    #   Col 1 = Period End    ("01/2026") ← use for quarter key
    #   Col 2 = EPS actual    ("1.62")
    #   Col 3 = EPS forecast  ("/1.52")  ← strip leading "/"
    #   Col 4 = Revenue actual ("68.1B")
    #   Col 5 = Rev forecast  ("/65.56B") ← strip leading "/"
    #   Col 6 = EPS Surprise %
    #   Col 7 = Revenue Surprise %
    if soup is not None:
        # Find earnings table: the one with 'Period End' and 'Revenue' headers
        target_table = None
        for tbl in soup.find_all("table"):
            ths = [th.get_text(strip=True) for th in tbl.find_all("th")]
            if "Period End" in ths and "Revenue" in ths:
                target_table = tbl
                break
        # Fallback: second table on page (index 1) which is the earnings table
        if target_table is None:
            all_tables = soup.find_all("table")
            if len(all_tables) > 1:
                target_table = all_tables[1]

        if target_table:
            COL_PERIOD_END = 1
            COL_EPS_ACT    = 2
            COL_EPS_EST    = 3
            COL_REV_ACT    = 4
            COL_REV_EST    = 5

            def _parse_rev_cell(cell_text: str) -> Optional[float]:
                t = cell_text.strip().lstrip("/").replace(",", "").replace(" ", "")
                if not t or t in ("-", "N/A", "", "--"): return None
                mult = 1.0
                if t[-1].upper() == "T": mult = 1e12; t = t[:-1]
                elif t[-1].upper() == "B": mult = 1e9;  t = t[:-1]
                elif t[-1].upper() == "M": mult = 1e6;  t = t[:-1]
                elif t[-1].upper() == "K": mult = 1e3;  t = t[:-1]
                try: return float(t) * mult
                except: return None

            def _parse_eps_cell(cell_text: str) -> Optional[float]:
                t = cell_text.strip().lstrip("/").replace(",", "")
                if not t or t in ("-", "N/A", "", "--"): return None
                try: return float(t)
                except: return None

            def _parse_period_end(cell_text: str) -> Optional[str]:
                """'04/2026' → '2026-04'"""
                t = cell_text.strip()
                if "/" in t:
                    parts = t.split("/")
                    if len(parts) == 2:
                        try:
                            return f"{int(parts[1]):04d}-{int(parts[0]):02d}"
                        except Exception:
                            pass
                try:
                    import dateutil.parser as _dp
                    return _dp.parse(t).strftime("%Y-%m")
                except Exception:
                    return None

            tbody = target_table.find("tbody") or target_table
            for tr in tbody.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) < 6:
                    continue
                def _cell(col):
                    return cells[col].get_text(strip=True) if col < len(cells) else ""

                d = _parse_period_end(_cell(COL_PERIOD_END))
                if not d:
                    continue

                ic_by_qtr[d] = {
                    "eps_actual":   _parse_eps_cell(_cell(COL_EPS_ACT)),
                    "eps_estimate": _parse_eps_cell(_cell(COL_EPS_EST)),
                    "rev_actual":   _parse_rev_cell(_cell(COL_REV_ACT)),
                    "rev_estimate": _parse_rev_cell(_cell(COL_REV_EST)),
                }

    # ── 5. Merge ic_by_qtr into earnings_dates rows ──────────────────────
    if not ic_by_qtr:
        return 0

    filled = 0
    for row in earnings_dates:
        d = (row.get("date") or "")[:7]
        candidates = [d]
        try:
            y, m = int(d[:4]), int(d[5:7])
            for delta in [-1, -2, 1, 2]:
                nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                candidates.append(f"{ny:04d}-{nm:02d}")
        except Exception:
            pass
        for key in candidates:
            if key not in ic_by_qtr:
                continue
            q = ic_by_qtr[key]
            # EPS estimate
            if row.get("eps_estimate") is None and q.get("eps_estimate") is not None:
                row["eps_estimate"] = q["eps_estimate"]
                row["_eps_est_source"] = "investing_com"
                filled += 1
                est = _safe_float(row["eps_estimate"])
                rep = _safe_float(row.get("eps_reported"))
                if np.isfinite(est) and np.isfinite(rep) and abs(est) > 0.001:
                    diff = rep - est
                    row["eps_surprise_pct"] = round((diff / abs(est)) * 100.0, 2)
            # Revenue estimate
            if row.get("revenue_estimate") is None and q.get("rev_estimate") is not None:
                row["revenue_estimate"] = q["rev_estimate"]
                row["_rev_est_source"] = "investing_com"
                filled += 1
            # Revenue reported (consensus-tracked actual, different from income stmt)
            if row.get("revenue_reported") is None and q.get("rev_actual") is not None:
                row["revenue_reported"] = q["rev_actual"]
                row["_rev_act_source"] = "investing_com"
            break

    # ── 6. Extend quarterly_revenue with IC actuals (deeper history) ──────
    # investing.com covers ~16-20 quarters vs yfinance's hard 5-quarter cap.
    # We write IC revenue actuals into quarterly_revenue for quarters NOT already
    # present, ordered chronologically. YoY growth is recomputed after extension.
    if quarterly_revenue is not None and ic_by_qtr:
        existing_months = {(r.get("date") or "")[:7]
                           for r in quarterly_revenue if r.get("date")}
        new_rows = []
        for month_key, q in ic_by_qtr.items():
            rev_act = q.get("rev_actual")
            if rev_act is None or not np.isfinite(float(rev_act) if rev_act else float("nan")):
                continue
            # Skip if we already have this quarter from yfinance/FMP
            if month_key in existing_months:
                continue
            # Also skip if within ±1 month of an existing entry
            try:
                y, m = int(month_key[:4]), int(month_key[5:7])
                adjacent = False
                for delta in [-1, 1]:
                    nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                    if f"{ny:04d}-{nm:02d}" in existing_months:
                        adjacent = True
                        break
                if adjacent:
                    continue
            except Exception:
                pass
            new_rows.append({
                "date": f"{month_key}-01",   # first of month as date string
                "revenue": float(rev_act),
                "revenue_yoy_growth": None,
                "revenue_source": "investing_com",
            })

        if new_rows:
            # Merge and sort all rows chronologically
            combined = list(quarterly_revenue) + new_rows
            combined.sort(key=lambda r: (r.get("date") or ""))
            # Recompute YoY on the extended series (quarter i vs i-4)
            for i, row in enumerate(combined):
                if i >= 4:
                    prev = combined[i - 4].get("revenue")
                    curr = row.get("revenue")
                    if prev and prev > 0 and curr and np.isfinite(curr) and np.isfinite(prev):
                        row["revenue_yoy_growth"] = round((curr / prev - 1.0) * 100.0, 2)
            # Update the list in-place: clear and re-extend
            quarterly_revenue.clear()
            quarterly_revenue.extend(combined)
            filled += len(new_rows)

    return filled


def fetch_fmp_single(yahoo_ticker: str, api_key: str) -> Dict[str, Any]:
    """Fetch earnings + revenue data for one ticker from FMP.

    Endpoints used (all available on free tier):
        /v3/income-statement/{symbol}?period=quarter&limit=12
        /v3/earnings-surprises/{symbol}

    Returns same dict shape as fetch_earnings_data() so the two sources
    are interchangeable in the results cache.
    """
    sym = _yahoo_to_fmp(yahoo_ticker)
    out: Dict[str, Any] = {
        "ticker": yahoo_ticker,
        "fmp_symbol": sym,
        "quarterly_revenue": [],
        "earnings_dates": [],
        "catalyst_events": [],
        "info": {},
        "data_source": "fmp",
    }

    # 1) Quarterly revenue from income statement
    try:
        stmt = _fmp_get("/income-statement", {"symbol": sym, "period": "quarter", "limit": 12}, api_key)
        if stmt and isinstance(stmt, list):
            rev_rows = []
            for q in reversed(stmt):  # FMP returns newest first — reverse for chronological
                rev = _safe_float(q.get("revenue"))
                date_str = str(q.get("date", ""))[:10]
                if rev and np.isfinite(rev):
                    rev_rows.append({"date": date_str, "revenue": rev, "revenue_yoy_growth": None})
            # Compute YoY
            for i, row in enumerate(rev_rows):
                if i >= 4:
                    prev = rev_rows[i - 4]["revenue"]
                    if prev and prev > 0:
                        row["revenue_yoy_growth"] = round((row["revenue"] / prev - 1.0) * 100.0, 2)
            out["quarterly_revenue"] = rev_rows
            # Populate info.revenue_growth from most recent YoY
            recent_yoy = next((r["revenue_yoy_growth"] for r in reversed(rev_rows)
                               if r["revenue_yoy_growth"] is not None), None)
            if recent_yoy is not None:
                out["info"]["revenue_growth"] = recent_yoy / 100.0  # normalise like yfinance
    except Exception as e:
        out["_fmp_rev_error"] = str(e)

    # 2) EPS beat/miss from earnings surprises endpoint
    # Also pull revenue estimates from FMP analyst-estimates (quarterly) to get
    # real revenue consensus beat/miss — this is what yfinance fails to return.
    try:
        surprises = _fmp_get("/earnings-surprises", {"symbol": sym}, api_key)
        if surprises and isinstance(surprises, list):
            eps_rows = []
            for s in surprises[:16]:  # last 16 quarters max
                est = _safe_float(s.get("estimatedEps") or s.get("epsEstimated"))
                act = _safe_float(s.get("actualEps") or s.get("actualEarningResult"))
                surprise_pct = None
                if est is not None and act is not None and np.isfinite(est) and np.isfinite(act):
                    if abs(est) > 0.001:
                        surprise_pct = round((act / est - 1.0) * 100.0, 2)
                date_str = str(s.get("date", ""))[:10]
                eps_rows.append({
                    "date": date_str,
                    "eps_estimate": est if (est is not None and np.isfinite(est)) else None,
                    "eps_reported": act if (act is not None and np.isfinite(act)) else None,
                    "eps_surprise_pct": surprise_pct,
                    "revenue_estimate": None,   # populated below from analyst-estimates
                    "revenue_reported": None,   # populated below from income-statement match
                    "_method": "fmp_earnings_surprises",
                })
            out["earnings_dates"] = eps_rows
    except Exception as e:
        out["_fmp_eps_error"] = str(e)

    # 2b) Revenue consensus estimates from FMP analyst-estimates endpoint
    # Matches estimates to eps_rows by YYYY-MM key, fills revenue_estimate + revenue_reported.
    # Free tier: available for US tickers. Non-US may return empty list.
    try:
        rev_est_data = _fmp_get("/analyst-estimates", {"symbol": sym, "period": "quarter", "limit": 16}, api_key)
        if rev_est_data and isinstance(rev_est_data, list):
            # Build YYYY-MM -> revenue consensus estimate
            rev_est_by_qtr = {}
            for q in rev_est_data:
                d = str(q.get("date", ""))[:7]
                avg = _safe_float(q.get("estimatedRevenueAvg") or q.get("revenueAvg"))
                if d and avg is not None and np.isfinite(avg) and avg > 0:
                    rev_est_by_qtr[d] = avg
            # Build YYYY-MM -> actual revenue from income statement (already fetched)
            rev_act_by_qtr = {}
            for qr in out.get("quarterly_revenue", []):
                d = (qr.get("date") or "")[:7]
                if d and qr.get("revenue") is not None:
                    rev_act_by_qtr[d] = qr["revenue"]
            # Populate revenue_estimate / revenue_reported in eps_rows
            filled = 0
            for row in out.get("earnings_dates", []):
                d = (row.get("date") or "")[:7]
                # Try exact match then ±2 month window
                candidates = [d]
                try:
                    y, m = int(d[:4]), int(d[5:7])
                    for delta in [-1, -2, 1, 2]:
                        nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                        candidates.append(f"{ny:04d}-{nm:02d}")
                except Exception: pass
                for key in candidates:
                    if row.get("revenue_estimate") is None and key in rev_est_by_qtr:
                        row["revenue_estimate"] = rev_est_by_qtr[key]
                        filled += 1
                    if row.get("revenue_reported") is None and key in rev_act_by_qtr:
                        row["revenue_reported"] = rev_act_by_qtr[key]
                    if row.get("revenue_estimate") is not None and row.get("revenue_reported") is not None:
                        break
            if filled:
                out["_fmp_rev_estimates_filled"] = filled
    except Exception as e:
        out["_fmp_rev_est_error"] = str(e)

    out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out


def fetch_fmp_batch(tickers: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """Fetch FMP data for a list of tickers with rate-limit awareness.

    Free tier: 250 calls/day → 2 calls per ticker (income-stmt + surprises)
    = ~125 tickers per day on free tier.
    Starter: 300 calls/min → effectively unlimited.

    Tickers are fetched in the order given (caller should pass FMP priority order).
    """
    results: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(tickers):
        if i > 0 and i % 50 == 0:
            print(f"[gc-fmp] progress: {i}/{len(tickers)}")
            time.sleep(1.0)
        try:
            data = fetch_fmp_single(t, api_key)
            results[t] = data
        except Exception as e:
            results[t] = {"ticker": t, "error": str(e), "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        time.sleep(0.20)  # ~5 tickers/sec = 10 calls/sec — safe for free and paid tiers
    return results


def fetch_earnings_universe(
    tickers: List[str],
    existing_cache: Dict[str, Any] = None,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch earnings data for the full universe with caching.

    Args:
        tickers: list of ticker symbols
        existing_cache: previously cached data (skip if fresh enough)
        force: if True, re-download everything regardless of cache

    Returns:
        dict of {ticker: earnings_data}
    """
    cache = existing_cache or {}
    now = dt.datetime.now(dt.timezone.utc)
    results: Dict[str, Dict[str, Any]] = {}
    to_fetch: List[str] = []

    dead_skipped = 0
    for t in tickers:
        t = str(t).strip()
        if not t or is_ghost_ticker(t):
            continue
        if t in KNOWN_DEAD_TICKERS:
            # Mark inactive in cache so downstream knows, but never fetch
            if t not in cache:
                cache[t] = {
                    "ticker": t, "inactive": True,
                    "inactive_since": now.isoformat(),
                    "inactive_reason": "known_dead_no_yahoo_support",
                    "quarterly_revenue": [], "earnings_dates": [],
                    "catalyst_events": [], "info": {}, "data_gap_alert": True,
                }
            results[t] = cache[t]
            dead_skipped += 1
            continue
        if t in cache:
            cached = cache[t]
            # Assign batch_day on first encounter (persisted in cache going forward)
            if "batch_day" not in cached:
                exch = t.rsplit(".", 1)[-1] if "." in t else "US"
                if exch != "US":
                    cached["batch_day"] = assign_batch_day(t)
            # Was previously rate-limited with no data? Always re-fetch.
            was_rate_limited = (
                "_info_error" in cached
                and not cached.get("info")
                and not cached.get("quarterly_revenue")
                and not cached.get("earnings_dates")
            )
            if not was_rate_limited and not _should_fetch_today(t, cached, now, force):
                results[t] = cached
                continue
        to_fetch.append(t)

    cached_count = len(results) - dead_skipped
    print(f"[gc-data] universe={len(tickers)}, cached={cached_count}, dead_skipped={dead_skipped}, to_fetch={len(to_fetch)}")

    # ── Exchange suffix helper ────────────────────────────────────
    def _exch(t: str) -> str:
        return t.rsplit(".", 1)[-1] if "." in t else "US"

    # ── Priority fetch order ──────────────────────────────────────
    # Tier 1: US + core EU (yfinance most reliable, largest universe)
    # Tier 2: Asia-Pacific + EM (yfinance works but patchier)
    # Within each tier, group by exchange to avoid cross-market throttle
    FETCH_ORDER = [
        # Tier 1 — US + EU
        "US",    # NYSE / NASDAQ (no suffix)
        "L",     # London Stock Exchange
        "DE",    # XETRA (Frankfurt)
        "PA",    # Euronext Paris
        "AS",    # Euronext Amsterdam
        "MI",    # Borsa Italiana
        "MC",    # Madrid
        "SW",    # SIX Swiss Exchange
        "ST",    # Stockholm
        "OL",    # Oslo
        "HE",    # Helsinki
        "CO",    # Copenhagen
        "TO",    # Toronto (TSX)
        # Tier 2 — Asia/Pacific + EM (FMP fallback applies here)
        "T",     # Tokyo (TSE)
        "KS",    # Korea Exchange
        "HK",    # Hong Kong
        "TW",    # Taiwan
        "AX",    # ASX (Australia)
        "NS",    # NSE India
        "BO",    # BSE India
        "SS",    # Shanghai
        "SZ",    # Shenzhen
        "SA",    # B3 Brazil
        "JO",    # JSE South Africa
        "MX",    # BMV Mexico
        "SR",    # Saudi Tadawul
        "IS",    # Istanbul (Borsa İstanbul)
        "WA",    # Warsaw
        "KL",    # Kuala Lumpur
        "BK",    # Thailand SET
        "JK",    # Jakarta IDX
        "PS",    # Philippine SE
        "QA",    # Qatar Exchange
        "AD",    # Abu Dhabi
        "DU",    # Dubai
        "AT",    # Athens
        "SA",    # B3 Brazil (also catches suffix-fixed Brazilian tickers)
        "SN",    # Santiago Chile
        "PR",    # Prague Czech Republic
        "BD",    # Budapest Hungary
        "CA",    # Cairo Egypt
        "KA",    # Karachi Pakistan
    ]

    # Exchanges where FMP fallback is activated after yfinance failure
    # Ordered by priority — UK/Korea/Japan first as requested
    FMP_PRIORITY_EXCHANGES = ["L", "KS", "T", "HK", "AX", "PA", "SW", "AS", "DE", "MI", "NS", "TW"]

    by_exchange: Dict[str, List[str]] = {}
    for t in to_fetch:
        ex = _exch(t)
        by_exchange.setdefault(ex, []).append(t)

    # Build ordered fetch list: FETCH_ORDER first, then any remaining exchanges alphabetically
    ordered_fetch: List[str] = []
    seen_exch = set()
    for ex in FETCH_ORDER:
        if ex in by_exchange:
            ordered_fetch.extend(by_exchange[ex])
            seen_exch.add(ex)
    for ex in sorted(by_exchange):
        if ex not in seen_exch:
            ordered_fetch.extend(by_exchange[ex])

    # ── First pass: yfinance ──────────────────────────────────────
    yf_failed: List[str] = []   # completely empty after all 4 methods

    for i, t in enumerate(ordered_fetch):
        if i > 0 and i % 100 == 0:
            print(f"[gc-data] progress: {i}/{len(ordered_fetch)} fetched")
            time.sleep(0.75)   # Hard pause every 100 — resets Yahoo rate-limit window
        try:
            data = fetch_earnings_data(t)
            results[t] = data
            has_past_eps = any(e.get("eps_reported") is not None for e in data.get("earnings_dates", []))
            has_rev = len(data.get("quarterly_revenue", [])) >= 4
            has_info = data.get("info", {}).get("revenue_growth") is not None
            if not has_past_eps and not has_rev and not has_info and "error" not in data:
                yf_failed.append(t)
        except Exception as e:
            results[t] = {"ticker": t, "error": str(e), "fetched_at": now.isoformat()}
            yf_failed.append(t)
        ex = _exch(t)
        pause = 0.22 if ex == "US" else 0.11
        if i % 5 == 4:
            time.sleep(pause)

    # ── yfinance retry pass (5-second cooldown) ───────────────────
    # Second attempt before involving FMP — covers transient throttle hits
    if yf_failed:
        print(f"[gc-data] yfinance retry: {len(yf_failed)} tickers empty on first pass")
        time.sleep(3.5)
        still_failed: List[str] = []
        for i, t in enumerate(yf_failed):
            if i > 0 and i % 30 == 0:
                time.sleep(1.5)
            try:
                data = fetch_earnings_data(t)
                results[t] = data
                has_past_eps = any(e.get("eps_reported") is not None for e in data.get("earnings_dates", []))
                has_rev = len(data.get("quarterly_revenue", [])) >= 4
                has_info = data.get("info", {}).get("revenue_growth") is not None
                if not has_past_eps and not has_rev and not has_info and "error" not in data:
                    still_failed.append(t)
            except Exception as e:
                results[t] = {"ticker": t, "error": str(e), "fetched_at": now.isoformat()}
                still_failed.append(t)
            time.sleep(0.30)
        yf_failed = still_failed

    # ── FMP fallback (if API key present) ────────────────────────
    # Strategy: only route FMP_TARGET_EXCHANGES to FMP — these are markets
    # where yfinance structurally fails. Other exchanges use FMP only if
    # truly empty. This maximises the 250 free calls/day on highest-ROI tickers.
    if yf_failed:
        fmp_key = os.environ.get("FMP_API_KEY", "").strip()
        if fmp_key:
            def _fmp_priority(t: str) -> int:
                ex = _exch(t)
                try:
                    return FMP_TARGET_EXCHANGES.index(ex)
                except ValueError:
                    return len(FMP_TARGET_EXCHANGES)

            # Prioritise structural gaps first, then any remaining empties
            fmp_queue = sorted(yf_failed, key=_fmp_priority)
            target_count = sum(1 for t in fmp_queue if _exch(t) in FMP_TARGET_EXCHANGES)
            print(f"[gc-data] FMP fallback: {len(fmp_queue)} tickers "
                  f"({target_count} in target exchanges: {','.join(FMP_TARGET_EXCHANGES[:5])}...)")
            fmp_results = fetch_fmp_batch(fmp_queue, fmp_key)
            for t, fdata in fmp_results.items():
                if fdata:
                    existing = results.get(t, {})
                    merged = {**existing, **fdata, "data_source": "fmp_fallback"}
                    if len(existing.get("quarterly_revenue", [])) >= len(fdata.get("quarterly_revenue", [])):
                        merged["quarterly_revenue"] = existing["quarterly_revenue"]
                    results[t] = merged
            fmp_ok = sum(1 for t in fmp_queue if results.get(t, {}).get("data_source") == "fmp_fallback")
            print(f"[gc-data] FMP recovered {fmp_ok}/{len(fmp_queue)} tickers")
        else:
            print(f"[gc-data] {len(yf_failed)} tickers still empty after yfinance retries. "
                  f"Set FMP_API_KEY env var to activate FMP fallback.")

    # ── Tag data gaps ─────────────────────────────────────────────
    # data_gap_alert = True means this ticker has NO usable earnings data.
    # Used by scan mode to flag when a technical signal cannot be confirmed
    # with Star 2/3 due to missing data (different from a genuine miss).
    for t, v in results.items():
        has_rev = len(v.get("quarterly_revenue", [])) >= 4
        has_eps = any(e.get("eps_reported") is not None for e in v.get("earnings_dates", []))
        has_info = bool(v.get("info", {}).get("revenue_growth"))
        v["data_gap_alert"] = not (has_rev or has_eps or has_info)

    # ── Summary ───────────────────────────────────────────────────
    ok = sum(1 for v in results.values() if "error" not in v)
    rev_ok = sum(1 for v in results.values() if len(v.get("quarterly_revenue", [])) >= 4)
    eps_ok = sum(1 for v in results.values() if any(e.get("eps_reported") for e in v.get("earnings_dates", [])))
    catalyst_ok = sum(1 for v in results.values() if v.get("catalyst_events"))
    fmp_count = sum(1 for v in results.values() if v.get("data_source") == "fmp_fallback")
    gap_count = sum(1 for v in results.values() if v.get("data_gap_alert"))
    print(
        f"[gc-data] done: {ok}/{len(results)} success | "
        f"revenue_data: {rev_ok} | eps_history: {eps_ok} | "
        f"catalyst_events: {catalyst_ok} | fmp_fallback: {fmp_count} | "
        f"data_gaps: {gap_count}"
    )

    return results


# ────────────────────────────────────────────────────────────────
# Revenue analytics (computed from cached earnings data)
# ────────────────────────────────────────────────────────────────
def compute_revenue_analytics(earnings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived revenue metrics for a single ticker.

    Returns:
        latest_yoy_growth: most recent quarter YoY revenue growth %
        prev_yoy_growth: prior quarter YoY growth %
        is_accelerating: latest > prev
        growth_streak: consecutive quarters of positive YoY growth
        accel_streak: consecutive quarters of acceleration
        meets_golden_momentum: revenue >= 20% YoY
        revenue_source: 'quarterly' (full data) or 'info_fallback' (TTM from yfinance info)
    """
    out: Dict[str, Any] = {}
    rev = earnings_data.get("quarterly_revenue", [])

    # ── Fallback: use info.revenue_growth when quarterly data is unavailable ──
    # Covers Japan (.T), UK (.L), Australia (.AX), France (.PA), Switzerland (.SW),
    # and most EM markets where yfinance doesn't return structured quarterly income
    # statements. info.revenue_growth is trailing-12-month YoY — good enough for
    # Layer 3 (>=20% gate) and sector comparisons. Flagged as 'info_fallback' so
    # star rating can apply a lower confidence weight if desired.
    if not rev or len(rev) < 5:
        info = earnings_data.get("info", {})
        info_rev_growth = info.get("revenue_growth")
        if info_rev_growth is not None:
            try:
                g = float(info_rev_growth)
                if np.isfinite(g):
                    pct = round(g * 100.0, 2)
                    out["latest_yoy_growth"] = pct
                    out["prev_yoy_growth"] = None
                    out["is_accelerating"] = None
                    out["growth_streak"] = 1 if g > 0 else 0
                    out["accel_streak"] = 0
                    out["meets_golden_momentum_revenue"] = pct >= 20.0
                    out["revenue_source"] = "info_fallback"
            except Exception:
                pass
        return out

    # Get quarters with YoY growth computed (need at least 2)
    with_growth = [r for r in rev if r.get("revenue_yoy_growth") is not None]
    if not with_growth:
        return out

    # Most recent quarter first (sort by date descending for latest)
    with_growth_sorted = sorted(with_growth, key=lambda r: r["date"], reverse=True)

    latest = with_growth_sorted[0]
    out["latest_revenue"] = latest.get("revenue")
    out["latest_revenue_date"] = latest.get("date")
    out["latest_yoy_growth"] = latest["revenue_yoy_growth"]

    if len(with_growth_sorted) >= 2:
        prev = with_growth_sorted[1]
        out["prev_yoy_growth"] = prev["revenue_yoy_growth"]
        out["is_accelerating"] = (
            latest["revenue_yoy_growth"] is not None
            and prev["revenue_yoy_growth"] is not None
            and latest["revenue_yoy_growth"] > prev["revenue_yoy_growth"]
        )
    else:
        out["prev_yoy_growth"] = None
        out["is_accelerating"] = None

    # Growth streak (how many consecutive quarters of positive YoY growth, from most recent)
    streak = 0
    for r in with_growth_sorted:
        g = r.get("revenue_yoy_growth")
        if g is not None and g > 0:
            streak += 1
        else:
            break
    out["growth_streak"] = streak

    # Acceleration streak (how many consecutive quarters where growth > prior quarter's growth)
    accel_streak = 0
    for i in range(len(with_growth_sorted) - 1):
        curr_g = with_growth_sorted[i].get("revenue_yoy_growth")
        prev_g = with_growth_sorted[i + 1].get("revenue_yoy_growth")
        if curr_g is not None and prev_g is not None and curr_g > prev_g:
            accel_streak += 1
        else:
            break
    out["accel_streak"] = accel_streak

    # Golden momentum: revenue >= 20% YoY
    out["meets_golden_momentum_revenue"] = (
        latest["revenue_yoy_growth"] is not None
        and latest["revenue_yoy_growth"] >= 20.0
    )
    out["revenue_source"] = "quarterly"

    return out


def compute_eps_analytics(earnings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute EPS beat/miss metrics.

    Returns:
        latest_eps_surprise: most recent quarter surprise %
        eps_beat_streak: consecutive quarters of positive surprise
        eps_miss_streak: consecutive quarters of negative surprise
        latest_eps_date: date of most recent earnings
        beat_revenue_and_eps: True if both revenue and EPS beat in latest quarter
    """
    out: Dict[str, Any] = {}
    dates = earnings_data.get("earnings_dates", [])
    if not dates:
        return out

    # Filter to past dates only (with reported EPS)
    past = [
        d for d in dates
        if d.get("eps_reported") is not None
        and d.get("date") is not None
        and d["date"] <= dt.date.today().isoformat()
    ]
    if not past:
        return out

    # Sort by date descending
    past_sorted = sorted(past, key=lambda r: r["date"], reverse=True)

    latest = past_sorted[0]
    out["latest_eps_date"] = latest["date"]
    out["latest_eps_estimate"] = latest.get("eps_estimate")
    out["latest_eps_reported"] = latest.get("eps_reported")
    out["latest_eps_surprise_pct"] = latest.get("eps_surprise_pct")

    # Compute surprise if not provided
    if out["latest_eps_surprise_pct"] is None:
        est = _safe_float(latest.get("eps_estimate"))
        rep = _safe_float(latest.get("eps_reported"))
        if np.isfinite(est) and np.isfinite(rep) and abs(est) > 0.001:
            out["latest_eps_surprise_pct"] = round((rep / est - 1.0) * 100.0, 2)

    # Check if revenue also beat (if data available)
    rev_est = _safe_float(latest.get("revenue_estimate"))
    rev_rep = _safe_float(latest.get("revenue_reported"))
    out["revenue_beat"] = (
        np.isfinite(rev_est) and np.isfinite(rev_rep)
        and rev_est > 0 and rev_rep > rev_est
    )
    eps_surprise = _safe_float(out.get("latest_eps_surprise_pct"))
    out["beat_revenue_and_eps"] = (
        out["revenue_beat"]
        and np.isfinite(eps_surprise)
        and eps_surprise > 0
    )

    # Build EPS YoY lookup: YYYY-MM -> eps_reported (for proxy when estimate unavailable)
    # Same-quarter-prior-year comparison: if this Q's EPS > Q-4's EPS, treat as "beat".
    # This is the last resort — only used when eps_estimate is still None after all
    # enrichment sources (investing.com, FMP, Finnhub) have been tried.
    eps_by_qtr: Dict[str, float] = {}
    for r in past_sorted:
        d = (r.get("date") or "")[:7]
        rep = _safe_float(r.get("eps_reported"))
        if d and np.isfinite(rep):
            eps_by_qtr[d] = rep

    def _eps_beat_for_row(r: Dict) -> Optional[bool]:
        """True/False/'match'/None.
        - True  = beat  (eps_reported > eps_estimate + $0.01 tolerance)
        - False = miss  (eps_reported < eps_estimate - $0.01 tolerance)
        - 'match' = in-line (within ±$0.01 of estimate) — streak-neutral
        - None  = undeterminable (no estimate + no usable proxy)

        Beat/miss is computed from eps_surprise_pct first (pre-computed by yfinance
        or enrichment sources), then from raw eps fields, then YoY proxy.
        The ±$0.01 match zone applies only when real estimates exist.
        YoY proxy (no estimate) returns True/False only — no match zone possible.
        """
        est = _safe_float(r.get("eps_estimate"))
        rep = _safe_float(r.get("eps_reported"))

        # --- Path A: real estimate available → apply ±$0.01 match zone ---
        if np.isfinite(est) and np.isfinite(rep):
            diff = rep - est
            if abs(diff) <= 0.01:
                return "match"
            return diff > 0  # True = beat, False = miss

        # --- Path B: pre-computed surprise_pct but no raw estimate ---
        # (surprise_pct from yfinance rounded; less precise — no match zone)
        s = r.get("eps_surprise_pct")
        if s is not None:
            return float(s) > 0

        # --- Path C: YoY proxy (compare to same quarter prior year) ---
        if np.isfinite(rep):
            d = (r.get("date") or "")[:7]
            try:
                y, m = int(d[:4]), int(d[5:7])
                py = y - 1
                for delta in [0, -1, 1, -2, 2]:
                    nm = m + delta; ny = py + (nm-1)//12; nm = ((nm-1)%12)+1
                    key = f"{ny:04d}-{nm:02d}"
                    if key in eps_by_qtr:
                        prior_eps = eps_by_qtr[key]
                        if prior_eps > 0.001:
                            result = rep > prior_eps
                            r["_eps_est_source"] = r.get("_eps_est_source", "yoy_proxy")
                            return result
            except Exception:
                pass
        return None  # truly cannot determine

    # EPS beat streak — 'match' (in-line) is streak-neutral: doesn't extend or break it
    beat_streak = 0
    miss_streak = 0
    for r in past_sorted:
        result = _eps_beat_for_row(r)
        if result is True:
            if miss_streak == 0:
                beat_streak += 1
            else:
                break
        elif result == "match":
            pass  # in-line: neither extends beat streak nor breaks it
        elif result is False:
            if beat_streak == 0:
                miss_streak += 1
            else:
                break  # beat streak ends on a genuine miss
        else:
            break  # None = undeterminable — stop streak
    out["eps_beat_streak"] = beat_streak
    out["eps_miss_streak"] = miss_streak

    # Human-readable result for the most recent quarter
    _r0 = past_sorted[0] if past_sorted else {}
    _latest_eps_result = _eps_beat_for_row(_r0)
    out["latest_eps_result"] = (
        "beat"  if _latest_eps_result is True   else
        "match" if _latest_eps_result == "match" else
        "miss"  if _latest_eps_result is False   else
        "unknown"
    )
    # Revenue result for most recent quarter (only meaningful when real estimate exists)
    _r0_rev_est = _safe_float(_r0.get("revenue_estimate"))
    _r0_rev_rep = _safe_float(_r0.get("revenue_reported"))
    if np.isfinite(_r0_rev_est) and np.isfinite(_r0_rev_rep) and _r0_rev_est > 0:
        _latest_rev_result_raw = _revenue_beat_for_row(_r0) if _r0 else None
        out["latest_rev_result"] = (
            "beat"  if _latest_rev_result_raw is True   else
            "match" if _latest_rev_result_raw == "match" else
            "miss"  if _latest_rev_result_raw is False   else
            "unknown"
        )
    else:
        out["latest_rev_result"] = "no_estimate"

    # Revenue + EPS dual-beat streak: consecutive quarters where BOTH
    # EPS AND revenue beat.
    #
    # Revenue beat priority:
    #   1) Strict: revenue_reported > revenue_estimate from earnings_dates
    #      (yfinance often lacks these — frequently None even for US large-caps)
    #   2) Proxy: quarterly_revenue YoY growth > 0 for that period
    #      (from income statement — available for ~75% of universe incl. NVDA, LLY)
    #
    # Without the proxy, NVDA/LLY/CRDO all score 0 despite massive revenue beats,
    # because yfinance 1.2.0 no longer reliably returns Revenue Estimate column.

    # Build a lookup: YYYY-MM -> revenue_yoy_growth from quarterly_revenue (income stmt)
    rev_yoy_by_qtr: Dict[str, float] = {}
    for qr in earnings_data.get("quarterly_revenue", []):
        d = qr.get("date")
        g = qr.get("revenue_yoy_growth")
        if d and g is not None and np.isfinite(g):
            rev_yoy_by_qtr[d[:7]] = g

    def _revenue_beat_for_row(r: Dict) -> Optional[bool]:
        """True/False/'match'/None.
        - True  = beat  (revenue_reported > revenue_estimate by >0.5%)
        - False = miss  (revenue_reported < revenue_estimate by >0.5%)
        - 'match' = in-line (within ±0.5% of consensus estimate) — streak-neutral
        - None  = no consensus estimate available (cannot determine beat/miss)

        The YoY proxy (revenue grew vs prior year) is intentionally NOT used here.
        'Revenue grew YoY' ≠ 'beat consensus'. Without real estimates we return None
        rather than a misleading True/False.
        """
        r_est = _safe_float(r.get("revenue_estimate"))
        r_rep = _safe_float(r.get("revenue_reported"))
        if np.isfinite(r_est) and np.isfinite(r_rep) and r_est > 0:
            ratio = (r_rep - r_est) / r_est
            if abs(ratio) <= 0.005:    # ±0.5% = in-line / match
                return "match"
            return ratio > 0           # True = beat, False = miss
        return None  # no consensus estimate — undeterminable

    rev_beat_streak = 0
    for r in past_sorted:
        rev_beat = _revenue_beat_for_row(r)
        eps_beat = _eps_beat_for_row(r)
        # Both must be genuine beats (True). 'match' is neutral — doesn't extend
        # the streak but also doesn't break it. None or False breaks the streak.
        if rev_beat is True and eps_beat is True:
            rev_beat_streak += 1
        elif rev_beat == "match" and eps_beat in (True, "match"):
            pass  # in-line on both or one — neutral, don't count but don't break
        else:
            break
    out["revenue_beat_streak"] = rev_beat_streak

    return out


def compute_sector_medians(
    earnings_cache: Dict[str, Dict[str, Any]],
    universe_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute median YoY revenue growth per sector.
    Used for Layer 3 (Golden Momentum) — stock must outgrow sector median.
    """
    sector_growths: Dict[str, List[float]] = {}
    ticker_to_sector = dict(zip(
        universe_df["Ticker"].astype(str),
        universe_df["Sector"].astype(str),
    ))

    for ticker, data in earnings_cache.items():
        rev_analytics = compute_revenue_analytics(data)
        growth = rev_analytics.get("latest_yoy_growth")
        if growth is not None and np.isfinite(growth):
            sector = ticker_to_sector.get(ticker, "Unknown")
            sector_growths.setdefault(sector, []).append(growth)

    medians: Dict[str, float] = {}
    for sector, growths in sector_growths.items():
        if growths:
            medians[sector] = float(np.median(growths))

    return medians


# ────────────────────────────────────────────────────────────────
# Summary report (for inspection / debugging)
# ────────────────────────────────────────────────────────────────
def print_data_summary(
    earnings_cache: Dict[str, Dict[str, Any]],
    universe_df: pd.DataFrame,
) -> None:
    """Print a summary of the earnings data quality and top growth stocks."""
    total = len(earnings_cache)
    has_rev = sum(1 for v in earnings_cache.values() if len(v.get("quarterly_revenue", [])) >= 4)
    has_eps = sum(1 for v in earnings_cache.values() if len(v.get("earnings_dates", [])) >= 1)
    has_error = sum(1 for v in earnings_cache.values() if "error" in v)

    print(f"\n{'=' * 60}")
    print(f"GC ENGINE — DATA LAYER SUMMARY (v{GC_VERSION})")
    print(f"{'=' * 60}")
    print(f"Universe: {total} tickers")
    print(f"Revenue data (>=4 quarters): {has_rev} ({has_rev/max(total,1)*100:.0f}%)")
    print(f"Earnings dates: {has_eps} ({has_eps/max(total,1)*100:.0f}%)")
    print(f"Errors: {has_error}")

    # Sector medians
    sector_medians = compute_sector_medians(earnings_cache, universe_df)
    if sector_medians:
        print(f"\nSector median YoY revenue growth:")
        for sector, median in sorted(sector_medians.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sector:35s} {median:+.1f}%")

    # Top growth stocks (golden momentum candidates)
    print(f"\nTop 20 stocks by YoY revenue growth (golden momentum candidates):")
    growth_list = []
    for ticker, data in earnings_cache.items():
        rev = compute_revenue_analytics(data)
        eps = compute_eps_analytics(data)
        g = rev.get("latest_yoy_growth")
        if g is not None and np.isfinite(g):
            growth_list.append({
                "ticker": ticker,
                "yoy_growth": g,
                "accel": rev.get("is_accelerating"),
                "growth_streak": rev.get("growth_streak", 0),
                "eps_beat_streak": eps.get("eps_beat_streak", 0),
                "revenue_beat_streak": eps.get("revenue_beat_streak", 0),
                "golden": rev.get("meets_golden_momentum_revenue", False),
                "rev_source": rev.get("revenue_source", "none"),
            })
    growth_list.sort(key=lambda x: x["yoy_growth"], reverse=True)

    print(f"\n**Top 20 by Rev YoY** *(golden momentum candidates)*:\n")
    print(f"| Ticker | Rev YoY | Accel | Growth Streak | EPS Beat |")
    print(f"| :-- | --: | :--: | --: | --: |")
    for r in growth_list[:20]:
        accel = "✓" if r["accel"] else ""
        star = "⭐⭐⭐" if r["golden"] else ""
        print(f"| {r['ticker']} {star} | {r['yoy_growth']:+.1f}% | {accel} "
              f"| {r['growth_streak']}Q | {r['eps_beat_streak']}Q |")

    # ── Data coverage diagnostic (markdown tables) ─────────────────────────────
    # Counts three distinct data types separately to avoid confusion:
    #   1. Revenue ACTUALS   — from quarterly_income_stmt  → quarterly_revenue field
    #   2. Revenue ESTIMATES — analyst consensus forecasts  → revenue_estimate in earnings_dates
    #   3. EPS ESTIMATES     — analyst consensus forecasts  → eps_estimate in earnings_dates
    # These live in different fields and come from different sources!

    # Per-ticker tallies keyed by source
    _SOURCES = ["yfinance", "investing_com", "fmp", "finnhub", "yoy_proxy", "none"]
    eps_est_src:    Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_act_src:    Dict[str, int] = {s: 0 for s in _SOURCES}  # income-stmt actuals
    rev_est_src:    Dict[str, int] = {s: 0 for s in _SOURCES}  # consensus estimates
    rev_con_src:    Dict[str, int] = {s: 0 for s in _SOURCES}  # consensus-reported (FMP/IC/FH)

    ic_filled_total = fmp_enrich_total = finnhub_filled_total = 0

    for t_key, data in earnings_cache.items():
        past_ed = [r for r in data.get("earnings_dates", [])
                   if r.get("eps_reported") is not None]

        # ── EPS estimate source ──────────────────────────────────────────────
        if any(r.get("eps_estimate") is not None for r in past_ed):
            srcs = [r.get("_eps_est_source", "yfinance")
                    for r in past_ed if r.get("eps_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            eps_est_src[dom if dom in eps_est_src else "yfinance"] += 1
        elif any(r.get("_eps_est_source") == "yoy_proxy" for r in past_ed):
            eps_est_src["yoy_proxy"] += 1
        else:
            eps_est_src["none"] += 1

        # ── Revenue ACTUALS from income stmt (quarterly_revenue) ────────────
        # Source attribution: what fetched the quarterly_revenue for this ticker?
        qr = [r for r in data.get("quarterly_revenue", []) if r.get("revenue") is not None]
        if qr:
            qr_src = data.get("data_source", "yfinance")   # "fmp" or "yfinance"
            rev_act_src[qr_src if qr_src in rev_act_src else "yfinance"] += 1
        else:
            rev_act_src["none"] += 1

        # ── Revenue ESTIMATES (consensus analyst forecasts) ─────────────────
        if any(r.get("revenue_estimate") is not None for r in past_ed):
            srcs = [r.get("_rev_est_source", "yfinance")
                    for r in past_ed if r.get("revenue_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            rev_est_src[dom if dom in rev_est_src else "yfinance"] += 1
        else:
            rev_est_src["none"] += 1

        # ── Revenue CONSENSUS-REPORTED (what FMP/investing.com track as actuals) ─
        if any(r.get("revenue_reported") is not None for r in past_ed):
            srcs = [r.get("_rev_act_source", "yfinance")
                    for r in past_ed if r.get("revenue_reported") is not None]
            dom = max(set(srcs), key=srcs.count)
            rev_con_src[dom if dom in rev_con_src else "yfinance"] += 1
        else:
            rev_con_src["none"] += 1

        ic_filled_total      += data.get("_investing_com_filled", 0)
        fmp_enrich_total     += data.get("_fmp_rev_estimates_filled", 0)
        finnhub_filled_total += data.get("_finnhub_filled", 0)

    N = max(len(earnings_cache), 1)

    def _pn(n: int) -> str:
        """Format count + percentage: '2198 (86%)'"""
        return f"{n} ({n * 100 // N}%)"

    def _blank(n: int) -> str:
        return _pn(n) if n > 0 else "–"

    # ── Table 1: EPS Data Coverage ──────────────────────────────────────────
    print("\n**EPS data coverage** (per ticker, dominant source):\n")
    print(f"| | yfinance | investing.com | FMP | Finnhub | YoY proxy | none |")
    print(f"| :-- | --: | --: | --: | --: | --: | --: |")
    print(f"| Estimate "
          f"| {_blank(eps_est_src['yfinance'])} "
          f"| {_blank(eps_est_src['investing_com'])} "
          f"| {_blank(eps_est_src['fmp'])} "
          f"| {_blank(eps_est_src['finnhub'])} "
          f"| {_blank(eps_est_src['yoy_proxy'])} "
          f"| {_blank(eps_est_src['none'])} |")
    # EPS reported is always from yfinance (it's the actual EPS the company reported)
    eps_rep_count = sum(
        1 for d in earnings_cache.values()
        if any(r.get("eps_reported") is not None
               for r in d.get("earnings_dates", []))
    )
    print(f"| Reported (actual EPS) "
          f"| {_pn(eps_rep_count)} | – | – | – | – | {_pn(N - eps_rep_count)} |")

    # ── Table 2: Revenue Data Coverage ─────────────────────────────────────
    # Three distinct data types shown as separate rows
    print("\n**Revenue data coverage** (per ticker, dominant source):\n")
    print(f"| | yfinance | investing.com | FMP | Finnhub | none |")
    print(f"| :-- | --: | --: | --: | --: | --: |")

    # Row A: Actuals from income statement (quarterly_revenue)
    print(f"| **Actuals** *(income stmt, ≥1Q)* "
          f"| {_blank(rev_act_src['yfinance'])} "
          f"| {_blank(rev_act_src['investing_com'])} "
          f"| {_blank(rev_act_src['fmp'])} "
          f"| {_blank(rev_act_src['finnhub'])} "
          f"| {_blank(rev_act_src['none'])} |")

    # Row B: Consensus estimates (analyst forecasts — what we NEED for real beat/miss)
    print(f"| **Consensus Estimate** *(analyst forecast)* "
          f"| {_blank(rev_est_src['yfinance'])} "
          f"| {_blank(rev_est_src['investing_com'])} "
          f"| {_blank(rev_est_src['fmp'])} "
          f"| {_blank(rev_est_src['finnhub'])} "
          f"| {_blank(rev_est_src['none'])} |")

    # Row C: Consensus-reported (what FMP/investing.com/Finnhub surface as "reported" alongside their estimates)
    print(f"| **Consensus Reported** *(paired with estimate)* "
          f"| {_blank(rev_con_src['yfinance'])} "
          f"| {_blank(rev_con_src['investing_com'])} "
          f"| {_blank(rev_con_src['fmp'])} "
          f"| {_blank(rev_con_src['finnhub'])} "
          f"| {_blank(rev_con_src['none'])} |")

    # Note row explaining the cascade
    print(f"\n~ *Consensus estimates cascade*: yfinance → investing.com → FMP → Finnhub. "
          f"Without a consensus estimate, revenue beat/miss cannot be determined. "
          f"YoY proxy (actual revenue > 0 prior year) is **not** equivalent to beating consensus "
          f"and is no longer used for revenue_beat_streak.")

    if ic_filled_total or fmp_enrich_total or finnhub_filled_total:
        print(f"\n*Fields enriched this run* — "
              f"investing.com: {ic_filled_total} | "
              f"FMP: {fmp_enrich_total} | "
              f"Finnhub: {finnhub_filled_total}")

    hints = []
    if not os.environ.get("FINNHUB_API_KEY"):
        hints.append("`FINNHUB_API_KEY`")
    if not os.environ.get("FMP_API_KEY"):
        hints.append("`FMP_API_KEY`")
    if hints:
        print(f"\n> ⚠️ Set secrets to improve consensus estimate coverage: {', '.join(hints)}")
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n**Star gate proxy counts** *(data mode — no OHLCV, no OpenAI)*:\n")
    star2_proxy = [r for r in growth_list
                   if r["eps_beat_streak"] >= EPS_BEAT_STREAK_MIN
                   and r["revenue_beat_streak"] >= EPS_BEAT_STREAK_MIN]
    star2_eps_only = [r for r in growth_list
                      if r["eps_beat_streak"] >= EPS_BEAT_STREAK_MIN
                      and r["revenue_beat_streak"] < EPS_BEAT_STREAK_MIN]
    star3_proxy = [r for r in star2_proxy
                   if r["golden"]
                   and r["rev_source"] not in ("info_fallback", "annual_estimated", "none")]

    print(f"| Star | Criterion | Count | Note |")
    print(f"| :-- | :-- | --: | :-- |")
    print(f"| ⭐ | Technical ignition (OHLCV) | *scan only* | Not scored in data mode |")
    print(f"| ⭐⭐ | EPS beat + Rev beat ≥{EPS_BEAT_STREAK_MIN}Q | {len(star2_proxy)} | "
          f"+{len(star2_eps_only)} EPS-only (rev estimates missing) |")
    print(f"| ⭐⭐⭐ | Star 2 + last-Q rev ≥20% YoY | {len(star3_proxy)} | "
          f"Moat confirmation via OpenAI in scan mode |")

    if star3_proxy:
        star3_proxy.sort(key=lambda x: x["yoy_growth"], reverse=True)
        print(f"\n**⭐⭐⭐ Star 3 candidates** *(top {min(15, len(star3_proxy))})*:\n")
        print(f"| Ticker | Rev YoY | Accel | EPS Streak | Rev Beat Streak |")
        print(f"| :-- | --: | :--: | --: | --: |")
        for r in star3_proxy[:15]:
            accel = "✓" if r["accel"] else ""
            print(f"| {r['ticker']} | {r['yoy_growth']:+.1f}% | {accel} "
                  f"| {r['eps_beat_streak']}Q | {r['revenue_beat_streak']}Q |")
    else:
        print(f"\n*(No Star 3 candidates — revenue_beat_streak requires consensus revenue estimates)*")

    # Catalyst events summary (Layer-2 non-earnings triggers)
    print(f"\nTier-1 catalyst events detected (last 180 days):")
    catalyst_list = []
    for ticker, data in earnings_cache.items():
        for ev in data.get("catalyst_events", []):
            if ev.get("catalyst_tier") == 1:
                catalyst_list.append({
                    "ticker": ticker,
                    "date": ev.get("date", ""),
                    "type": ev.get("catalyst_type", ""),
                    "headline": ev.get("headline", "")[:80],
                })
    catalyst_list.sort(key=lambda x: x["date"], reverse=True)
    for c in catalyst_list[:20]:
        print(f"  {c['ticker']:12s} [{c['type']:12s}] {c['date']}  {c['headline']}")
    if not catalyst_list:
        print("  (none found — catalyst scan requires news to be recent and indexed)")
    print()


# ────────────────────────────────────────────────────────────────
# Step 2: Ignition detection (Phase 0)
# ─────────────────────────────────────────────────────────────────────────────
# OpenAI helpers — Star 2 catalyst confirmation + Star 3 moat assessment
# Both functions fall back gracefully when OPENAI_API_KEY is absent or call fails.
# Using gpt-4o-mini: fast, cheap, good enough for binary yes/no + 1-sentence rationale.
# ─────────────────────────────────────────────────────────────────────────────

def _openai_chat(prompt: str, max_tokens: int = 100) -> Optional[str]:
    """
    Shared helper: single OpenAI chat/completions call.
    Returns the raw assistant message text, or None on any failure.
    """
    import urllib.request as _req, json as _json, os as _os
    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    body = _json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    try:
        req = _req.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with _req.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        raw = data["choices"][0]["message"]["content"].strip()
        # strip markdown code fences if model wraps output
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        return raw
    except Exception as e:
        print(f"  [openai] call failed: {e}")
        return None


def _openai_catalyst_is_massive(headline: str, company: str) -> Dict[str, Any]:
    """
    Ask OpenAI whether a catalyst headline is a genuinely company-thesis-changing
    event (Star 2 equivalent without earnings data).

    Examples that qualify: FDA approval, $1B+ government contract, war-driven demand surge.
    Examples that do NOT: routine partnerships, minor product updates, analyst upgrades.

    Returns:
        confirmed  bool | None  — True=massive, False=not massive, None=API unavailable
        rationale  str          — one-sentence reasoning
    """
    prompt = (
        f"Company: {company}\n"
        f"News headline: \"{headline}\"\n\n"
        "Is this a MASSIVE, company-thesis-changing catalyst? "
        "Qualifying events: FDA approval, >$500M government/defense contract, "
        "geopolitical shock that directly reshapes demand (e.g. war, sanctions), "
        "transformative M&A (company doubles in size).\n"
        "Non-qualifying: routine partnerships, analyst upgrades, minor product launches, "
        "earnings beats (those are scored separately).\n\n"
        "Respond ONLY as JSON: {\"massive\": true/false, \"rationale\": \"one sentence\"}"
    )
    raw = _openai_chat(prompt, max_tokens=80)
    if raw is None:
        return {"confirmed": None, "rationale": "API unavailable"}
    try:
        import json as _json
        parsed = _json.loads(raw)
        return {
            "confirmed": bool(parsed.get("massive")),
            "rationale": str(parsed.get("rationale", "")).strip(),
        }
    except Exception:
        return {"confirmed": None, "rationale": f"Parse error: {raw[:60]}"}


def _openai_moat_assessment(
    ticker: str,
    short_name: str,
    sector: str,
    industry: str,
    yoy_growth: float,
    eps_beat_streak: int,
    revenue_beat_streak: int,
) -> Dict[str, Any]:
    """
    Ask OpenAI whether a Star 3 candidate has a durable economic moat —
    i.e. does it do what it does significantly better than any direct competitor,
    protected by structural advantages (network effects, switching costs,
    cost advantages, intangible assets, or efficient scale).

    Think Netflix/Amazon circa 2005 — structural lead, not just a good quarter.

    Returns:
        moat_confirmed  bool | None
        moat_rationale  str
        moat_source     str
    """
    prompt = (
        f"Company: {short_name} ({ticker})\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"Revenue YoY growth: {yoy_growth:.1f}%\n"
        f"EPS beat streak: {eps_beat_streak}Q | Revenue beat streak: {revenue_beat_streak}Q\n\n"
        "Does this company have a DURABLE ECONOMIC MOAT — structural advantages that make it "
        "significantly better than any direct competitor and hard to displace? "
        "Think network effects, switching costs, proprietary IP, cost scale, or brand. "
        "A fast-growing company without a moat (e.g. commodity producer, cyclical) does NOT qualify.\n\n"
        "Respond ONLY as JSON: {\"moat\": true/false, \"rationale\": \"one sentence max 20 words\"}"
    )
    raw = _openai_chat(prompt, max_tokens=80)
    if raw is None:
        return {"moat_confirmed": None, "moat_rationale": "API unavailable — moat not assessed", "moat_source": "fallback"}
    try:
        import json as _json
        parsed = _json.loads(raw)
        return {
            "moat_confirmed": bool(parsed.get("moat")),
            "moat_rationale": str(parsed.get("rationale", "")).strip(),
            "moat_source": "gpt-4o-mini",
        }
    except Exception:
        return {"moat_confirmed": None, "moat_rationale": f"Parse error: {raw[:60]}", "moat_source": "fallback"}


# All 4 criteria must hold for MIN_IGNITION_SESSIONS consecutive bars.
#   I1: daily price move >= 0.5 × ATR(14)
#   I2: CLV >= +0.70  (buyers dominating top 15% of range)
#   I3: volume >= 2.0 × AvgVol(20)  (institutional footprint)
#   I4: cumulative move over ignition window >= 1.5 × ATR(14)
# ────────────────────────────────────────────────────────────────
MIN_IGNITION_SESSIONS = 3
I1_ATR_MULT   = 0.5
I2_CLV_MIN    = 0.70
I3_VOL_MULT   = 2.0
I4_CUM_MULT   = 1.5


def detect_ignition(
    ticker: str,
    df: pd.DataFrame,
    earnings_cache: Dict[str, Any],
    n_sessions: int = MIN_IGNITION_SESSIONS,
    lookback_bars: int = 30,
) -> Dict[str, Any]:
    """Detect Phase 0 ignition signal for a single ticker.

    Returns dict with:
        star1          bool   — I1-I4 met for n_sessions consecutive bars
        stars          int    — total star rating (1=ignition, 2=+perf, 3=+golden)
        ticker         str
        ignition_start_date  str
        consecutive_sessions int
        cumulative_move_atr_ratio float
        yoy_growth     float  — latest quarterly YoY revenue %
        eps_beat_streak int
        data_gap_alert bool   — True = technically triggered but no earnings data
    """
    out: Dict[str, Any] = {"ticker": ticker, "star1": False, "stars": 0}

    if df is None or len(df) < 25:
        return out

    atr_s   = atr(df)
    avg_vol = df["Volume"].rolling(20).mean()

    # Evaluate I1–I3 bar by bar
    bar_flags: List[bool] = []
    for i in range(len(df)):
        if i < 20:
            bar_flags.append(False)
            continue
        atr_val = _safe_float(atr_s.iloc[i])
        vol_avg = _safe_float(avg_vol.iloc[i])
        if not (np.isfinite(atr_val) and atr_val > 0 and np.isfinite(vol_avg) and vol_avg > 0):
            bar_flags.append(False)
            continue

        prev_close = _safe_float(df["Close"].iloc[i - 1])
        curr_close = _safe_float(df["Close"].iloc[i])
        price_move = abs(curr_close - prev_close)
        clv_val    = clv_at_bar(df, i)
        vol        = _safe_float(df["Volume"].iloc[i])

        i1 = price_move >= I1_ATR_MULT * atr_val
        i2 = np.isfinite(clv_val) and clv_val >= I2_CLV_MIN
        i3 = np.isfinite(vol) and vol >= I3_VOL_MULT * vol_avg
        bar_flags.append(i1 and i2 and i3)

    # Scan last `lookback_bars` for n_sessions consecutive hits
    search_start = max(0, len(bar_flags) - lookback_bars)
    best_run_start: Optional[int] = None
    best_run_len   = 0
    run_start: Optional[int] = None
    run_len = 0

    for i in range(search_start, len(bar_flags)):
        if bar_flags[i]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= n_sessions and run_len >= best_run_len:
                best_run_len   = run_len
                best_run_start = run_start
        else:
            run_len = 0
            run_start = None

    if best_run_start is None or best_run_len < n_sessions:
        return out

    # I4: cumulative price move over the ignition window
    i4_start  = best_run_start
    i4_end    = best_run_start + best_run_len - 1
    atr_val   = _safe_float(atr_s.iloc[i4_start])
    start_px  = _safe_float(df["Close"].iloc[i4_start - 1])
    end_px    = _safe_float(df["Close"].iloc[i4_end])
    cum_move  = abs(end_px - start_px)
    cum_ratio = round(cum_move / atr_val, 2) if atr_val > 0 else 0.0
    i4_met    = cum_ratio >= I4_CUM_MULT

    if not i4_met:
        return out

    # ── Star 1 confirmed ─────────────────────────────────────────
    out["star1"] = True
    out["stars"] = 1
    out["ignition_start_date"]        = str(df.index[i4_start].date())
    out["consecutive_sessions"]       = best_run_len
    out["cumulative_move_atr_ratio"]  = cum_ratio

    # ── Earnings data from cache ──────────────────────────────────
    edata = earnings_cache.get(ticker, {})
    out["data_gap_alert"] = edata.get("data_gap_alert", True)

    rev_analytics = compute_revenue_analytics(edata)
    eps_analytics = compute_eps_analytics(edata)

    out["yoy_growth"]            = rev_analytics.get("latest_yoy_growth")
    out["latest_revenue_date"]   = rev_analytics.get("latest_revenue_date")   # date of last quarterly report
    out["is_accelerating"]       = rev_analytics.get("is_accelerating")
    out["eps_beat_streak"]       = eps_analytics.get("eps_beat_streak", 0)
    out["revenue_beat_streak"]   = eps_analytics.get("revenue_beat_streak", 0)
    out["revenue_source"]        = rev_analytics.get("revenue_source", "none")

    # ── Star 2: BOTH EPS + Revenue beats ≥2Q in a row OR massive catalyst ──
    # Dual-beat: company must have beaten consensus on BOTH top-line and
    # bottom-line for at least 2 consecutive quarters.
    # Catalyst path: a truly company-thesis-changing event (FDA, major contract,
    # geopolitical demand shock) confirmed by OpenAI — not just a keyword match.
    eps_streak  = out["eps_beat_streak"]
    rev_streak  = out["revenue_beat_streak"]
    meets_dual_beat = (eps_streak >= EPS_BEAT_STREAK_MIN and rev_streak >= EPS_BEAT_STREAK_MIN)

    # Check for massive catalyst (OpenAI-confirmed)
    massive_catalyst = False
    catalyst_rationale = ""
    for ev in edata.get("catalyst_events", []):
        if ev.get("catalyst_tier") == 1:
            info_block = edata.get("info") or {}
            ai_result = _openai_catalyst_is_massive(
                headline=ev.get("headline", ""),
                company=info_block.get("short_name", ticker),
            )
            if ai_result.get("confirmed"):
                massive_catalyst = True
                catalyst_rationale = ai_result.get("rationale", "")
                ev["ai_confirmed_massive"] = True
                ev["ai_rationale"] = catalyst_rationale
                break
            else:
                ev["ai_confirmed_massive"] = False

    out["massive_catalyst"] = massive_catalyst
    out["catalyst_rationale"] = catalyst_rationale

    # Star 2 data-gap fallback:
    # If we exhausted all 4 sources (yfinance, investing.com, FMP, Finnhub) and
    # still can't confirm BOTH beats, allow EITHER EPS>=2Q OR revenue>=2Q to
    # proceed to Star 3 assessment. This prevents good companies from being
    # knocked out due to missing consensus data, not actual performance.
    # The Star 3 moat + 20% YoY gate is the real quality filter anyway.
    has_any_rev_consensus = any(
        r.get("revenue_estimate") is not None
        for r in edata.get("earnings_dates", [])
        if r.get("eps_reported") is not None
    )
    data_gap_single = (
        not meets_dual_beat
        and not massive_catalyst
        and not has_any_rev_consensus        # truly no revenue data from any source
        and eps_streak >= EPS_BEAT_STREAK_MIN  # at least EPS beat confirmed
    )
    meets_rev_only = (rev_streak >= EPS_BEAT_STREAK_MIN and eps_streak < EPS_BEAT_STREAK_MIN)

    star2 = meets_dual_beat or massive_catalyst or data_gap_single or meets_rev_only
    if out["data_gap_alert"]:
        out["star2_blocked"] = "no_data"
    elif star2:
        out["stars"] = 2
        if meets_dual_beat:
            out["star2_via"] = "dual_beat"
        elif massive_catalyst:
            out["star2_via"] = "catalyst"
        elif data_gap_single:
            out["star2_via"] = "data_gap_eps_only"   # flagged — EPS beat confirmed, rev data unavailable
        else:
            out["star2_via"] = "data_gap_rev_only"   # flagged — rev growth confirmed, EPS data weak

    # ── Star 3: Golden Momentum — Rev ≥20% YoY + Moat confirmed ──
    # Requires Star 2. Revenue must be growing ≥20% YoY (structural momentum,
    # not a one-off). Moat confirmed by OpenAI: company must do what it does
    # significantly better than any competitor (network effects, switching costs,
    # proprietary IP, cost scale). Think Netflix/Amazon in 2005.
    yoy = out.get("yoy_growth")
    rev_source = out.get("revenue_source", "none")
    # Star 3 requires the 20% growth to be from the LAST QUARTERLY EARNINGS REPORT,
    # not from FY/TTM info.revenueGrowth. Exclude info_fallback and annual_estimated.
    quarterly_yoy_only = rev_source not in ("info_fallback", "annual_estimated", "none")
    meets_rev20 = (
        yoy is not None and np.isfinite(yoy) and yoy >= 20.0
        and quarterly_yoy_only
    )

    if out["stars"] == 2 and meets_rev20:
        info_block = edata.get("info") or {}
        moat_result = _openai_moat_assessment(
            ticker              = ticker,
            short_name          = info_block.get("short_name", ticker),
            sector              = info_block.get("sector", ""),
            industry            = info_block.get("industry", ""),
            yoy_growth          = yoy,
            eps_beat_streak     = eps_streak,
            revenue_beat_streak = rev_streak,
        )
        out["moat_confirmed"]  = moat_result.get("moat_confirmed")
        out["moat_rationale"]  = moat_result.get("moat_rationale", "")
        out["moat_source"]     = moat_result.get("moat_source", "fallback")

        if moat_result.get("moat_confirmed"):
            out["stars"] = 3
            print(f"  [★★★] {ticker}: rev={yoy:.1f}%  moat={out['moat_rationale'][:70]}")
        else:
            print(f"  [★★ ] {ticker}: rev={yoy:.1f}% — moat NOT confirmed: {out['moat_rationale'][:70]}")

    return out


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Growth Compounder Engine")
    ap.add_argument("--mode", choices=["data", "scan", "backtest"], default="data",
                    help="data=download earnings, scan=detect ignitions, backtest=20yr backtest")
    ap.add_argument("--force", action="store_true",
                    help="Force re-download all earnings data (ignore cache)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit universe size (for testing)")
    args = ap.parse_args()

    print(f"[gc] Growth Compounder Engine v{GC_VERSION} — mode={args.mode}")

    # Load universe
    universe_df = load_universe()
    tickers = sorted(universe_df["Ticker"].astype(str).unique().tolist())
    if args.limit > 0:
        tickers = tickers[:args.limit]
    print(f"[gc] Universe (MSCI World + EM): {len(tickers)} tickers")

    if args.mode == "data":
        # Load existing cache
        state = load_gc_state()
        existing_cache = state.get("earnings_cache", {})

        # ── Universe reconciliation ───────────────────────────────
        # Runs every time so the cache stays in sync with MSCI CSV changes.
        # New tickers (just added to MSCI): flagged as new_constituent=True
        #   → force-fetched immediately regardless of batch day
        # Removed tickers (left MSCI): marked inactive=True in cache
        #   → retained for 90 days (signals may still be active) then pruned
        live_set  = set(tickers)
        cache_set = set(existing_cache.keys())
        now_utc   = dt.datetime.now(dt.timezone.utc)

        # New constituents — mark for immediate fetch
        new_tickers = live_set - cache_set
        if new_tickers:
            print(f"[gc] {len(new_tickers)} new MSCI constituents — will force-fetch today")
        for t in new_tickers:
            existing_cache[t] = {"ticker": t, "new_constituent": True}

        # Removed constituents — mark inactive, prune after 90 days
        removed_tickers = cache_set - live_set
        pruned = 0
        for t in removed_tickers:
            entry = existing_cache[t]
            if entry.get("inactive"):
                # Already flagged — check if 90 days old
                flagged_at = entry.get("inactive_since", "")
                try:
                    age_days = (now_utc - dt.datetime.fromisoformat(flagged_at)).days
                    if age_days > 90:
                        del existing_cache[t]
                        pruned += 1
                except Exception:
                    pass
            else:
                entry["inactive"] = True
                entry["inactive_since"] = now_utc.isoformat()
        if removed_tickers:
            print(f"[gc] {len(removed_tickers)} tickers no longer in MSCI universe "
                  f"({pruned} pruned after 90d, rest retained)")

        # Fetch earnings data
        earnings_cache = fetch_earnings_universe(
            tickers,
            existing_cache=existing_cache,
            force=args.force,
        )

        # Save to state
        state["earnings_cache"] = earnings_cache
        state["last_data_update"] = dt.datetime.now(dt.timezone.utc).isoformat()
        state["gc_version"] = GC_VERSION
        save_gc_state(state)
        print(f"[gc] State saved to {GC_STATE_PATH}")

        # Print summary
        print_data_summary(earnings_cache, universe_df)

    elif args.mode == "scan":
        # ── Step 2: Ignition detection ────────────────────────────
        state = load_gc_state()
        earnings_cache = state.get("earnings_cache", {})
        if not earnings_cache:
            print("[gc] No earnings cache found — run --mode data first")
            raise SystemExit(1)

        print(f"[gc] Running ignition scan on {len(tickers)} tickers...")

        # Download OHLCV for full universe in chunks
        import yfinance as yf
        signals: List[Dict[str, Any]] = []
        chunk_size = 50
        all_data: Dict[str, pd.DataFrame] = {}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            if i > 0 and i % 200 == 0:
                print(f"[gc-scan] OHLCV progress: {i}/{len(tickers)}")
                time.sleep(2.0)
            try:
                raw = yf.download(
                    chunk, period=DOWNLOAD_PERIOD, interval=DOWNLOAD_INTERVAL,
                    group_by="ticker", auto_adjust=True, progress=False, threads=True
                )
                for t in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw.copy()
                        else:
                            df = raw[t].copy() if t in raw.columns.get_level_values(0) else pd.DataFrame()
                        df = df.dropna(how="all")
                        if not df.empty:
                            all_data[t] = df
                    except Exception:
                        pass
            except Exception as e:
                print(f"[gc-scan] chunk download error: {e}")
            time.sleep(0.5)

        print(f"[gc-scan] OHLCV downloaded for {len(all_data)}/{len(tickers)} tickers")

        # Run ignition detection on each ticker
        for t in tickers:
            df = all_data.get(t)
            if df is None or len(df) < 25:
                continue
            sig = detect_ignition(t, df, earnings_cache)
            if sig.get("star1"):
                signals.append(sig)

        # Sort by stars desc, then yoy growth desc
        signals.sort(key=lambda x: (-(x.get("stars", 0)), -(x.get("yoy_growth") or 0)))

        # Save signals to state
        state["ignition_signals"] = signals
        state["last_scan_ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_gc_state(state)

        # Print summary
        print(f"\n[gc-scan] Found {len(signals)} ignition signals")
        three_star = [s for s in signals if s.get("stars", 0) >= 3]
        two_star   = [s for s in signals if s.get("stars", 0) == 2]
        one_star   = [s for s in signals if s.get("stars", 0) == 1]
        print(f"  ★★★ {len(three_star)}  ★★ {len(two_star)}  ★ {len(one_star)}")
        print()
        for s in signals[:30]:
            stars = "★" * s.get("stars", 1)
            gap   = " ⚠ NO-DATA" if s.get("data_gap_alert") else ""
            print(
                f"  {stars:3s} {s['ticker']:12s} "
                f"sessions={s.get('consecutive_sessions','?')}  "
                f"cum_atr={s.get('cumulative_move_atr_ratio','?')}x  "
                f"rev_yoy={s.get('yoy_growth','?')}%  "
                f"eps_beats={s.get('eps_beat_streak','?')}Q"
                f"{gap}"
            )

    elif args.mode == "backtest":
        print("[gc] Backtest mode not yet implemented (Step 6)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
