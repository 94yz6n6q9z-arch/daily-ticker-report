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
GC_VERSION = "0.3.0"

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
)


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
    """Method 2: tk.quarterly_earnings — separate endpoint, more reliable for US tickers.
    Returns EPS actual/estimate per quarter. No revenue beat data but EPS streak still computable."""
    qe = tk.quarterly_earnings
    if qe is None or qe.empty:
        return []
    rows = []
    for idx, row in qe.iterrows():
        actual = _safe_float(row.get("Earnings"))
        estimate = _safe_float(row.get("Estimate") if "Estimate" in row else row.get("EPS Estimate"))
        surprise_pct = None
        if np.isfinite(actual) and np.isfinite(estimate) and abs(estimate) > 0.001:
            surprise_pct = round((actual / estimate - 1.0) * 100.0, 2)
        rows.append({
            "date": str(pd.Timestamp(idx).date()) if hasattr(idx, 'date') else str(idx),
            "eps_estimate": estimate if np.isfinite(estimate) else None,
            "eps_reported": actual if np.isfinite(actual) else None,
            "eps_surprise_pct": surprise_pct,
            "revenue_estimate": None,
            "revenue_reported": None,
            "_method": "quarterly_earnings",
        })
    return rows


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
                    "revenue_estimate": None,
                    "revenue_reported": None,
                    "_method": "fmp_earnings_surprises",
                })
            out["earnings_dates"] = eps_rows
    except Exception as e:
        out["_fmp_eps_error"] = str(e)

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
        time.sleep(0.25)  # ~4 tickers/sec = 8 calls/sec — safe for free and paid tiers
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

    for t in tickers:
        t = str(t).strip()
        if not t or is_ghost_ticker(t):
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

    cached_count = len(results)
    print(f"[gc-data] universe={len(tickers)}, cached={cached_count}, to_fetch={len(to_fetch)}")

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
            time.sleep(1.0)   # Hard pause every 100 — resets Yahoo rate-limit window
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
        pause = 0.3 if ex == "US" else 0.15
        if i % 5 == 4:
            time.sleep(pause)

    # ── yfinance retry pass (5-second cooldown) ───────────────────
    # Second attempt before involving FMP — covers transient throttle hits
    if yf_failed:
        print(f"[gc-data] yfinance retry: {len(yf_failed)} tickers empty on first pass")
        time.sleep(5.0)
        still_failed: List[str] = []
        for i, t in enumerate(yf_failed):
            if i > 0 and i % 30 == 0:
                time.sleep(2.0)
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
            time.sleep(0.4)
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

    # Beat streak
    beat_streak = 0
    miss_streak = 0
    for r in past_sorted:
        s = r.get("eps_surprise_pct")
        if s is None:
            est = _safe_float(r.get("eps_estimate"))
            rep = _safe_float(r.get("eps_reported"))
            if np.isfinite(est) and np.isfinite(rep) and abs(est) > 0.001:
                s = (rep / est - 1.0) * 100.0
        if s is not None and s > 0:
            if miss_streak == 0:
                beat_streak += 1
            else:
                break
        elif s is not None and s < 0:
            if beat_streak == 0:
                miss_streak += 1
            else:
                break
        else:
            break
    out["eps_beat_streak"] = beat_streak
    out["eps_miss_streak"] = miss_streak

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
                "golden": rev.get("meets_golden_momentum_revenue", False),
            })
    growth_list.sort(key=lambda x: x["yoy_growth"], reverse=True)
    for r in growth_list[:20]:
        accel = "ACC" if r["accel"] else "   "
        golden = "★★★" if r["golden"] else "   "
        print(f"  {golden} {r['ticker']:12s} Rev YoY: {r['yoy_growth']:+7.1f}%  {accel}  "
              f"Growth streak: {r['growth_streak']}Q  EPS beat streak: {r['eps_beat_streak']}Q")

    # Stocks with consecutive earnings beats + strong revenue
    print(f"\nStocks with >={EPS_BEAT_STREAK_MIN} consecutive EPS beats AND revenue >= 20% YoY:")
    stars = [r for r in growth_list if r["eps_beat_streak"] >= EPS_BEAT_STREAK_MIN and r["golden"]]
    stars.sort(key=lambda x: x["yoy_growth"], reverse=True)
    for r in stars[:15]:
        accel = "ACCEL" if r["accel"] else "     "
        print(f"  ★★★ {r['ticker']:12s} Rev YoY: {r['yoy_growth']:+7.1f}%  {accel}  "
              f"EPS beats: {r['eps_beat_streak']}Q")
    if not stars:
        print("  (none found)")

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

    out["yoy_growth"]       = rev_analytics.get("latest_yoy_growth")
    out["is_accelerating"]  = rev_analytics.get("is_accelerating")
    out["eps_beat_streak"]  = eps_analytics.get("eps_beat_streak", 0)
    out["revenue_source"]   = rev_analytics.get("revenue_source", "none")

    # ── Star 2: Performance Validation ───────────────────────────
    # Earnings: latest EPS beat OR catalyst event tier 1
    has_eps_beat    = eps_analytics.get("latest_eps_beat", False)
    has_catalyst    = any(
        e.get("catalyst_tier") == 1
        for e in edata.get("catalyst_events", [])
    )
    star2 = has_eps_beat or has_catalyst
    if out["data_gap_alert"]:
        out["star2_blocked"] = "no_data"   # signal to report: cannot confirm
    elif star2:
        out["stars"] = 2

    # ── Star 3: Golden Momentum ───────────────────────────────────
    # Revenue >= 20% YoY AND moat (AI assessment deferred — flag for now)
    yoy = out.get("yoy_growth")
    meets_rev = yoy is not None and np.isfinite(yoy) and yoy >= 20.0
    if out["stars"] == 2 and meets_rev:
        out["stars"] = 3   # moat check deferred until AI layer implemented
        out["moat_check_pending"] = True

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
