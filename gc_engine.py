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
GC_VERSION = "0.1.0"
BASE_DIR = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parent))
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR = BASE_DIR / "docs"
GC_STATE_PATH = DOCS_DIR / "gc_state.json"
MSCI_CSV = CONFIG_DIR / "msci_world_classification.csv"

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
    """Load MSCI World classification CSV.
    Returns DataFrame with columns: Ticker, Company, Country, Sector
    """
    if not MSCI_CSV.exists():
        print(f"[gc] MSCI CSV not found at {MSCI_CSV}")
        return pd.DataFrame(columns=["Ticker", "Company", "Country", "Sector"])
    try:
        df = pd.read_csv(MSCI_CSV, dtype=str)
        df["Ticker"] = df["Ticker"].astype(str).str.strip()
        df = df[df["Ticker"].str.len() > 0]
        return df
    except Exception as e:
        print(f"[gc] Error loading MSCI CSV: {e}")
        return pd.DataFrame(columns=["Ticker", "Company", "Country", "Sector"])


# ────────────────────────────────────────────────────────────────
# Earnings data download
# ────────────────────────────────────────────────────────────────
def fetch_earnings_data(ticker: str) -> Dict[str, Any]:
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
        if not t:
            continue
        # Check cache freshness
        if not force and t in cache:
            cached = cache[t]
            fetched_at = cached.get("fetched_at")
            if fetched_at:
                try:
                    ts = pd.Timestamp(fetched_at)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    age_hours = (now - ts).total_seconds() / 3600.0
                    if age_hours < EARNINGS_CACHE_TTL_HOURS:
                        results[t] = cached
                        continue
                except Exception:
                    pass
        to_fetch.append(t)

    cached_count = len(results)
    print(f"[gc-data] universe={len(tickers)}, cached={cached_count}, to_fetch={len(to_fetch)}")

    # Fetch in batches with progress
    for i, t in enumerate(to_fetch):
        if i > 0 and i % 50 == 0:
            print(f"[gc-data] progress: {i}/{len(to_fetch)} fetched")
            time.sleep(0.5)  # Gentle throttle every 50 tickers
        try:
            data = fetch_earnings_data(t)
            results[t] = data
        except Exception as e:
            results[t] = {"ticker": t, "error": str(e), "fetched_at": now.isoformat()}
        # Brief pause to avoid rate limiting
        if i % 5 == 4:
            time.sleep(0.2)

    # Summary
    ok = sum(1 for v in results.values() if "error" not in v)
    rev_ok = sum(1 for v in results.values() if len(v.get("quarterly_revenue", [])) >= 4)
    eps_ok = sum(1 for v in results.values() if len(v.get("earnings_dates", [])) >= 1)
    print(f"[gc-data] done: {ok}/{len(results)} success | "
          f"revenue_data: {rev_ok} | earnings_dates: {eps_ok}")

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
    """
    out: Dict[str, Any] = {}
    rev = earnings_data.get("quarterly_revenue", [])
    if not rev or len(rev) < 5:
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
    print(f"\nStocks with >=3 consecutive EPS beats AND revenue >= 20% YoY:")
    stars = [r for r in growth_list if r["eps_beat_streak"] >= 3 and r["golden"]]
    stars.sort(key=lambda x: x["yoy_growth"], reverse=True)
    for r in stars[:15]:
        accel = "ACCEL" if r["accel"] else "     "
        print(f"  ★★★ {r['ticker']:12s} Rev YoY: {r['yoy_growth']:+7.1f}%  {accel}  "
              f"EPS beats: {r['eps_beat_streak']}Q")
    if not stars:
        print("  (none found)")
    print()


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
    print(f"[gc] Universe: {len(tickers)} tickers")

    if args.mode == "data":
        # Load existing cache
        state = load_gc_state()
        existing_cache = state.get("earnings_cache", {})

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
        print("[gc] Scan mode not yet implemented (Step 2)")

    elif args.mode == "backtest":
        print("[gc] Backtest mode not yet implemented (Step 6)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
