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

from universe import (
    load_universe,
    is_ghost_ticker,
    KNOWN_DEAD_TICKERS,
    TICKER_OVERRIDES,
    BASE_DIR,
    CONFIG_DIR,
    DOCS_DIR,
    GC_STATE_PATH,
    # Exchange & market classification
    DEAD_MARKET_SUFFIXES,
    EU_SUFFIXES,
    MIN_MCAP_US_EU,
    MIN_MCAP_OTHER,
    mcap_threshold,
    FMP_ALPHA_BATCH_SUFFIXES,
    ADR_MAP,
    get_fmp_symbol,
)

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
GC_VERSION = "0.8.1"

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
    "0.5.1": (
        "FMP endpoint updated to /stable/earnings (Starter plan compatible). "
        "Fixed UnboundLocalError in compute_eps_analytics (_revenue_beat_for_row "
        "called before definition). Both fixes applied 2026-03-10."
    ),
    "0.5.2": (
        "Fix 1 — Revenue linkage (step 5d): universal quarterly_revenue → earnings_dates "
        "revenue_reported linkage. Recovers revenue_reported for ~1,233 tickers that had "
        "EPS data but no revenue because income statement data was stored in quarterly_revenue "
        "but never linked back to earnings_dates rows. Skips annual_estimated (÷4 proxy) rows "
        "to avoid corrupting beat/miss scoring. Uses ±2 month date matching, tagged yf_income_stmt. "
        "Fix 2 — Method 4 annual fallback: _fetch_eps_method4 now tries quarterly_income_stmt "
        "first then falls back to annual income_stmt, recovering EPS + revenue_reported for ~236 "
        "tickers in annual-reporting markets (JO, AX, PA, SW, L) where yfinance has no quarterly "
        "earnings data. Revenue extracted directly from the income_stmt DataFrame, not the ÷4 proxy. "
        "Fix 3 — FMP fetch_fmp_single: replaced /analyst-estimates (402 on Starter plan) with "
        "/stable/earnings which fills eps_estimate + revenue_estimate + revenue_reported for "
        "non-US tickers fetched via the FMP fallback path."
    ),
    "0.5.3": (
        "Email coverage tables fix: print_data_summary() and _build_coverage_html() both "
        "rewritten to emit proper HTML <table> blocks. Root cause: md.append(f'| row |\\n') "
        "with trailing \\n, combined with '\\n'.join(md), created double newlines between table "
        "rows — the markdown `tables` extension treats any blank line as a paragraph break, "
        "rendering raw pipe characters instead of HTML tables in the email. "
        "Both functions now emit <table>/<tr>/<th>/<td> directly. "
        "Per-country coverage table expanded: adds EPS-est-only / Rev-est-only / No-est columns "
        "so blind spots are visible per market. Estimate gap table added showing top offending "
        "countries. Companion version: scan.py v97, update_msci_world_classification.py 1.6.0."
    ),
    "0.5.4": (
        "load_universe() wired to all 9 country CSVs: Japan (.T), Taiwan (.TW), China (.SS/.SZ), "
        "Hong Kong (.HK), Saudi Arabia (.SR), New Zealand (.NZ) added alongside existing "
        "World, EM, Korea. Each CSV loaded with graceful skip if file absent (first-run safe). "
        "Dedup priority fixed: World=1 > country-specific=2 > EM=3 — previous alphabetical sort "
        "accidentally put EM above World for duplicates. Priority now explicit via _SOURCE_PRIORITY. "
        "Companion: update_msci_world_classification.py 1.9.0, sync_msci_manual_tickers.py 1.0.0."
    ),
    "0.5.5": (
        "Modular architecture: universe concern extracted to universe.py (new file). "
        "Removed from gc_engine: all MSCI CSV path constants, _SOURCE_PRIORITY, TICKER_OVERRIDES, "
        "KNOWN_DEAD_TICKERS, _GHOST_PATTERN, is_ghost_ticker(), load_universe(). "
        "gc_engine now imports all of the above from universe.py. "
        "scan.py also updated to import from universe.py, eliminating its duplicate _is_ghost_ticker(). "
        "Companion: universe.py 1.0.0, scan.py v98."
    ),
    "0.6.0": (
        "FMP global expansion + universe size management. "
        "Fix 1 — FMP enrichment sym (partially reverted in 0.6.1): see 0.6.1 notes. "
        "Fix 2 — Dead-market skip: exchanges KL (Malaysia), PS (Philippines), AD (UAE Abu Dhabi) "
        "confirmed to return 0 data. Added DEAD_MARKET_SUFFIXES constant; tickers on these "
        "exchanges are skipped before any API call, same as KNOWN_DEAD_TICKERS. "
        "Fix 3 — Market-cap floor filter: universe trimmed from ~6,200 to ~4,200 active tickers. "
        "US + EU (MIN_MCAP_US_EU = $2B): suffixes US, L, DE, PA, AS, MI, MC, ST, OL, HE, CO, "
        "LS, BR, AT, IR, SW, WA. All other markets (MIN_MCAP_OTHER = $5B): EM, APAC, MENA. "
        "Filter uses cached info.market_cap; new tickers fetched once then filtered next run. "
        "Fix 4 — investing.com disabled: confirmed 0 hits in production (GitHub Actions IPs "
        "blocked). Call removed from hot path; function retained for local dev use."
    ),
    "0.6.2": (
        "Architecture: exchange classification constants moved to universe.py. FMP sym fix correction + alpha-batch expansion. "
        "Architecture (v0.6.2): DEAD_MARKET_SUFFIXES, EU_SUFFIXES, MIN_MCAP_US_EU, MIN_MCAP_OTHER, mcap_threshold(), FMP_ALPHA_BATCH_SUFFIXES moved from gc_engine.py to universe.py. Both gc_engine and scan.py now import from universe.py — single source of truth for all exchange classification. _EU_DM_BATCH_SUFFIXES (narrow EU-only list) replaced by FMP_ALPHA_BATCH_SUFFIXES (all alpha-bare-symbol exchanges globally: +India, Brazil, Mexico, Turkey, Indonesia, South Africa, Chile, Qatar, Kuwait, Singapore, adding ~900 more tickers to the FMP rev-missing batch). "        "Fix 1 — 5b sym revert: v0.6.0 changed sym_fmp to _yahoo_to_fmp(ticker) which was wrong "
        "for the /stable/earnings endpoint. That endpoint is US-centric and only matches bare "
        "symbols — passing ASML.AS or SAP.DE returns empty results silently. Reverted to "
        "ticker.split('.')[0].upper() (bare symbol) for the 5b block specifically. "
        "UK tickers work because _yahoo_to_fmp(.L) already strips .L (same result). "
        "EU large caps with US cross-listings now recover: ASML.AS→ASML (NASDAQ), SAP.DE→SAP (NYSE). "
        "Fix 2 — EU/DM rev-missing FMP batch: new post-yfinance pass for EU + DM tickers "
        "(.L, .DE, .PA, .AS, .SW, .MI, .MC, .ST, .OL, .HE, .CO, .WA, .TO, .AX) that have EPS "
        "estimates but no revenue estimates. Uses fetch_fmp_single() with bare symbol — runs "
        "~387 tickers through /income-statement + /earnings-surprises + /stable/earnings. "
        "Recovers revenue estimates for ASML, SAP, LVMH, Schneider, Adidas, Allianz, etc. "
        "APAC local-only markets excluded (numeric tickers like 2330.TW→2330 won't match in FMP). "
        "Note for scan.py: coverage table still shows all gc_state entries including "
        "below_min_mcap ones — scan.py needs updating to filter these for accurate count display."
    ),
    "0.6.3": (
        "ADR map for numeric APAC tickers. "
        "Added ADR_MAP dict + get_fmp_symbol() in universe.py. "
        "ADR_MAP has 25 entries: 2330.TW→TSM, 7203.T→TM, 6758.T→SONY, "
        "0700.HK→TCEHY, 9988.HK→BABA, 9618.HK→JD, 9999.HK→NTES, "
        "8035.T→TOELY, 6857.T→AVANF, 9984.T→SFTBY, 9432.T→NTTYY, "
        "7974.T→NTDOY, 9433.T→KDDIY, 4519.T→CHGCY, 6954.T→FANUY, "
        "8001.T→ITOCY, 4063.T→SHECY, 6501.T→HTHIY, 6367.T→DAIIF, "
        "2303.TW→UMC, 3711.TW→ASX, 2382.HK→SMPRY, 3690.HK→MPNGF. "
        "5b FMP block now calls get_fmp_symbol(ticker) instead of bare split. "
        "Samsung (005930.KS) + SK Hynix (000660.KS) intentionally excluded: "
        "no liquid US ADR — OTC stubs have no meaningful analyst coverage."
    ),
    "0.6.4": (
        "Fix pipeline ordering: yfinance before paid sources. "
        "Architecture rule: free data first, paid only for genuine gaps. "
        "Moved block 5d (quarterly_revenue→earnings_dates linkage) to run BEFORE "
        "5b (FMP) and 5c (Finnhub). Previously 5d ran last, so FMP won the race "
        "for revenue_reported on ~1,580 tickers where yfinance already had the same "
        "number in quarterly_revenue (verified: values match to the dollar). "
        "After fix: ~85% of revenue_reported comes from yf_income_stmt (free), "
        "FMP fills only the genuine ~8% gap where yfinance has no income statement. "
        "No change to data quality — same numbers, correct source attribution, "
        "and significantly fewer FMP API calls consumed per run."
    ),
    "0.6.5": (
        "Architecture: revenue_reported linkage moved from Phase B into Phase A. "
        "It is a pure in-memory join on data already fetched — no network call. "
        "Now runs immediately after out['earnings_dates'] is set, before any paid source. "
        "Fix _fetch_revenue_estimates_yahoo: replaced bare requests.get() (blocked on "
        "GitHub Actions — 0 production hits ever) with tk._data.get_raw_json() which "
        "uses yfinance's authenticated crumb/cookie session. "
        "earningsTrend data now stored in out['forward_estimates'] (not merged into "
        "past earnings_dates rows — upcoming quarter dates never match past rows). "
        "forward_estimates is the seed for estimates snapshot repository: each daily "
        "run captures current consensus; over time becomes historical estimate data."
    ),
    "0.6.6": (
        "Fix _fetch_revenue_estimates_yahoo: three bugs. "
        "(1) Guard was `revenue_estimate OR revenue_reported` — filtered out earningsTrend rows "
        "with eps_estimate only (no revenue consensus). Fixed to include all trend rows. "
        "(2) All rows were tagged _method=yahoo_quotesummary with no distinction between "
        "earningsTrend (forward) and earningsHistory (past). Added _is_forward=True/False. "
        "(3) Block 2c used exact date match for upcoming row enrichment. Fixed to ±2 months. "
        "forward_estimates now filtered to _is_forward=True only (genuine upcoming quarters). "
        "_yahoo_qs_fwd_rev counter added: rows in forward_estimates that have rev_estimate."
    ),
    "0.6.7": (
        "Fix _fetch_revenue_estimates_yahoo: remove handle_404=True from get_raw_json call. "
        "In yfinance 1.2.0 that kwarg does not exist — throws TypeError silently caught by "
        "except Exception, returning [] for every ticker. Root cause of 0% earningsTrend "
        "coverage for all non-US markets (India, Korea, Taiwan, Japan, HK). Confirmed by "
        "test_earningstrend.py: Strategy A (no handle_404) hits 100% for all broken country "
        "groups; 28/29 tickers pass. Single 404 failure (LVMH.PA genuinely absent on Yahoo) "
        "is correctly handled by the existing except block."
    ),
    "0.6.8": (
        "Five fixes + parallel fetch (later reverted — see 0.6.9). "
        "(Issue #4/#5) has_eps summary counter fixed: now counts tickers with at least one "
        "eps_reported row (was counting any earnings_dates row including forward-only). "
        "(Issue #6) FMP EU/DM rev-missing batch now logs date-match failures per run. "
        "(Issue #8) earningsHistory EPS estimates backfilled into past earnings_dates rows "
        "where eps_estimate was None. Tagged _eps_est_source='yahoo_earnings_history'. "
        "Recovers non-US EPS estimate coverage (India/Korea/Taiwan/Japan had 0% before). "
        "(Issue #10) Parallel fetch with ThreadPoolExecutor(max_workers=4) added — "
        "reverted in 0.6.9 because shared yfinance crumb session is invalidated by "
        "concurrent requests within ~2 minutes, causing 88% 401 failure rate. "
        "Removed batch_day scheduling: all tickers now fetch every weekday."
    ),
    "0.6.9": (
        "Fix 1 — Revert to serial fetch: ThreadPoolExecutor removed. "
        "Root cause confirmed: 4 parallel workers share a single yfinance crumb/cookie "
        "session. Yahoo invalidates the crumb within ~2 min of parallel load, causing "
        "88%+ of tickers to return empty (4,513/5,152 failed in both 11-Mar runs). "
        "Serial sequential loop restored (same as pre-0.6.8). 0.75s pause every 100 "
        "tickers maintained to respect Yahoo rate limits. "
        "Fix 2 — Cache preservation on failed fetch: if a refetch returns empty data "
        "(no eps_reported, no quarterly_revenue) AND the existing cache has good data, "
        "the old cache entry is preserved instead of being overwritten with empty. "
        "Tagged with _used_cached=True and _cache_fallback_run_date so downstream "
        "reporting can distinguish fresh vs cached data. yf_failed list is NOT "
        "extended for cache-preserved tickers (FMP fallback skipped — data already good). "
        "Fix 3 — Early abort on yfinance degradation: after the first 200 tickers "
        "are processed, if >50% are empty (>=100 failures), the remaining fetch is "
        "aborted. Remaining tickers pull from cache if available, or are left with an "
        "error=fetch_aborted_yf_degraded marker. This prevents force runs from wiping "
        "the entire state when Yahoo is rate-limiting. A prominent warning is logged. "
        "New staleness banner in _build_coverage_html: when any tickers are serving "
        "from cached data, a red warning table is prepended to the coverage HTML showing "
        "count/pct stale, oldest cache date, and the degradation run date."
    ),
    "0.7.0": (
        "Fix A — _last_known_mcap: market cap is now persisted as a top-level field "
        "independent of the info{} block. Written on every fresh fetch that returns a "
        "valid mcap > 0. Copied forward when cache-preservation (Fix 2) is triggered. "
        "Falls back from info.market_cap to _last_known_mcap in the mcap floor filter "
        "so the filter remains operative even after a degraded run wiped info fields. "
        "Fix B — FMP company-profile mcap fallback: after the FMP earnings pass, any "
        "active ticker still missing both info.market_cap and _last_known_mcap gets "
        "a lightweight FMP /stable/company-profile call to retrieve mktCap. Only runs "
        "when FMP_API_KEY is set and ticker exchange is in FMP_ALPHA_BATCH_SUFFIXES. "
        "Fix C — export_universe.py: corrected 14 wrong exchange-to-country mappings."
    ),
    "0.8.0": (
        "FX conversion + mcap re-validation + floor reduction. Companion: universe.py 1.5.0. "
        "(1) _CURRENCY_TO_YAHOO_FX: added KWF→KWD alias (yfinance reports Kuwait as KWF not KWD; "
        "1 KWD ≈ $3.27 — was treated as 1:1 causing 3 Kuwait tickers to be wrongly below_min_mcap). "
        "Added ZAC→ZAR alias with ×0.01 scale (South Africa cents). "
        "Added ILA→ILS alias with ×0.01 scale (Israeli agorot — .TA stocks showed mcap 100× too large). "
        "(2) _FMP_SUFFIX_MAP: expanded from 19 → 50+ entries covering all active exchanges. "
        "Added .IS (Turkey), .KQ (KOSDAQ), .AE (UAE unified→strip), .TWO (Taiwan Gretai→strip), "
        ".BO (India BSE→strip), .JK .BK .SN .SI .QA .KW .SR .TA .NZ etc. "
        "(3) mcap floor: $2B → $1B USD across all markets (universe.py MIN_MCAP_US_EU/OTHER). "
        "(4) below_min_mcap weekly re-validation: previously once a ticker was flagged below floor "
        "it was frozen forever. Now gc_engine does a light yfinance info re-fetch every 7 days "
        "for below_min_mcap tickers — if mcap has grown past the floor the flag is cleared and "
        "the ticker re-enters the full fetch pipeline automatically. _mcap_recheck_date field "
        "tracks the last check date in gc_state.json. "
        "(5) export_universe.py v4: _normalize_ticker applied in _read_csv_tickers() — fixes "
        "100+ phantom no_cache entries (AKBNK.E.IS etc. now correctly match AKBNK.IS in cache). "
        "(6) scan.py v102: below_min_mcap removed from OHLCV Option A filter — mcap gating "
        "is GC-only; scan.py fetches OHLCV for all universe tickers regardless of mcap."
    ),
    "0.8.1": (
        "mcap subdivision bug-fix: _mcap_to_usd() was applying the ZAc/ILA subdivision_scale "
        "(x0.01) to marketCap, but yfinance always returns marketCap in the base currency "
        "(ZAR / ILS) regardless of the price currency reported. This caused all 27 .JO and "
        "11 .TA tickers to have mcap understated 100x, wrongly flagging every South African "
        "and Israeli stock as below_min_mcap (e.g. Naspers stored as $0.43B instead of $42.8B, "
        "NICE Systems as $0.07B instead of $7.4B). Fix: _mcap_to_usd() now remaps ZAc->ZAR and "
        "ILA->ILS before the FX lookup, so the subdivision_scale in _get_fx_rate_to_usd is never "
        "applied to marketCap. The x0.01 scale remains correct for price conversions. "
        "GBp was already handled correctly via the pre-existing GBp->GBP mapping. "
        "Net effect: 38 wrongly-expelled tickers (27 .JO + 11 .TA) return to active universe."
    ),
}

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
# Ghost ticker filter — drop from universe before any yfinance call
# Defined in universe.py — imported above as is_ghost_ticker()
# KNOWN_DEAD_TICKERS and TICKER_OVERRIDES also imported from universe.py
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# Star 2 threshold — consecutive EPS beats required
# Lowered from 3 to 2 per spec revision 2026-03-08
# ────────────────────────────────────────────────────────────────
EPS_BEAT_STREAK_MIN = 2

# ────────────────────────────────────────────────────────────────
# Fetch scheduling
# All tickers (US + RoW) are fetched every weekday (Mon–Fri).
# Removed batch_day logic (v0.6.8): weekday batching is no longer
# needed now that each run processes the full universe sequentially.
# Serial fetch completes in ~2.5 hrs; cache skips non-stale tickers
# so cached daily runs are significantly faster.
# Earnings trigger: any ticker with earnings_date within 1 day is
#                   always fetched regardless of weekend/weekday.
# ────────────────────────────────────────────────────────────────


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
      1. force=True           → always fetch
      2. inactive constituent → never fetch (left MSCI universe)
      3. new_constituent      → always fetch immediately (just joined MSCI)
      4. earnings trigger     → fetch if earnings ±1 day (any day of week)
      5. all others           → fetch on weekdays (Mon–Fri) only
    """
    if force:
        return True
    if cached.get("inactive"):
        return False
    if cached.get("new_constituent"):
        return True
    if _has_earnings_today(cached, now.date()):
        return True
    return now.weekday() < 5  # Mon–Fri for everything


# ────────────────────────────────────────────────────────────────
# ── Exchange classification ──────────────────────────────────────────────
# DEAD_MARKET_SUFFIXES, EU_SUFFIXES, MIN_MCAP_US_EU, MIN_MCAP_OTHER,
# mcap_threshold(), FMP_ALPHA_BATCH_SUFFIXES — all imported from universe.py.
# Do NOT redefine here. universe.py is the single source of truth.

# ── FMP target exchanges ──────────────────────────────────────────────────────
# FMP target exchanges — yfinance structurally weak here.
# These exchanges get FMP fallback first before being marked empty.
# Ordered by coverage gap severity (worst first).
# ────────────────────────────────────────────────────────────────
FMP_TARGET_EXCHANGES = ["KL", "IS", "PS", "AD", "DU", "T", "AX", "L", "JO", "HK", "PA", "SW"]

# ── FX rate table for mcap USD conversion ────────────────────────────────────
# yfinance info["marketCap"] returns the value in LOCAL currency.
# We compare against a USD floor, so we must convert.
# All entries are "{CURRENCY}USD=X" format → Yahoo returns "1 CURRENCY = X USD".
_CURRENCY_TO_YAHOO_FX: Dict[str, str] = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPYUSD=X",
    "KRW": "KRWUSD=X", "TWD": "TWDUSD=X", "HKD": "HKDUSD=X",
    "AUD": "AUDUSD=X", "CAD": "CADUSD=X", "CHF": "CHFUSD=X",
    "SEK": "SEKUSD=X", "NOK": "NOKUSD=X", "DKK": "DKKUSD=X",
    "INR": "INRUSD=X", "BRL": "BRLUSD=X", "MXN": "MXNUSD=X",
    "SGD": "SGDUSD=X", "TRY": "TRYUSD=X", "THB": "THBUSD=X",
    "IDR": "IDRUSD=X", "ZAR": "ZARUSD=X", "QAR": "QARUSD=X",
    "KWD": "KWDUSD=X", "SAR": "SARUSD=X", "ILS": "ILSUSD=X",
    "PLN": "PLNUSD=X", "CLP": "CLPUSD=X", "COP": "COPUSD=X",
    "HUF": "HUFUSD=X", "CZK": "CZKUSD=X", "AED": "AEDUSD=X",
    "NZD": "NZDUSD=X", "EGP": "EGPUSD=X", "PKR": "PKRUSD=X",
    "BDT": "BDTUSD=X", "CNY": "CNYUSD=X", "CNH": "CNHUSD=X",
    # yfinance non-standard currency codes → map to ISO equivalent
    "KWF": "KWDUSD=X",  # Kuwait: yfinance reports "KWF" instead of ISO "KWD" (1 KWD ≈ $3.27)
    "ZAC": "ZARUSD=X",  # South Africa: yfinance reports cents "ZAC" — divided by 100 below
    # ILA (Israeli Agorot): yfinance reports "ILA" for .TA stocks.
    # 1 ILA = 0.01 ILS. We fetch ILSUSD=X and divide by 100 in _get_fx_rate_to_usd.
    "ILA": "ILSUSD=X",
}

# Module-level FX cache — populated once per process, reused across tickers.
_FX_RATE_CACHE: Dict[str, float] = {}


def _get_fx_rate_to_usd(currency: str) -> float:
    """Return the rate to multiply local-currency mcap by to get USD equivalent.

    Uses Yahoo Finance {CURRENCY}USD=X spot tickers.  Falls back to 1.0 if
    Yahoo is unavailable or the currency is unknown (conservative: keeps ticker
    in the pipeline rather than incorrectly filtering it out).

    Results are cached for the life of the process (one cache per gc_engine run).
    FMP company-profile mcap is already in USD — pass currency="USD" for those.
    """
    if not currency:
        return 1.0
    currency = currency.upper().strip()
    if currency in ("USD", "USX"):
        return 1.0
    if currency in _FX_RATE_CACHE:
        return _FX_RATE_CACHE[currency]

    fx_sym = _CURRENCY_TO_YAHOO_FX.get(currency)
    if not fx_sym:
        print(f"[fx] Unknown currency '{currency}' — assuming 1:1 (will not filter)")
        _FX_RATE_CACHE[currency] = 1.0
        return 1.0

    # Subdivision currencies: mcap is reported in sub-units, not the base currency.
    # ZAC = South African cents (1 ZAC = 0.01 ZAR), ILA = Israeli agorot (1 ILA = 0.01 ILS).
    # We fetch the base-currency USD rate then divide by 100.
    subdivision_scale = 0.01 if currency in ("ZAC", "ILA") else 1.0

    try:
        import yfinance as _yf
        info = _yf.Ticker(fx_sym).info
        rate = (info.get("regularMarketPrice")
                or info.get("previousClose")
                or info.get("bid"))
        if rate and float(rate) > 0:
            rate = float(rate) * subdivision_scale
            _FX_RATE_CACHE[currency] = rate
            print(f"[fx] {currency}: 1 {currency} = {rate:.6f} USD ({fx_sym}"
                  + ("×0.01)" if subdivision_scale != 1.0 else ")"))
            return rate
    except Exception as e:
        print(f"[fx] Could not fetch {fx_sym}: {e} — assuming 1:1")

    _FX_RATE_CACHE[currency] = 1.0
    return 1.0


def _mcap_to_usd(mcap_local: float, currency: str, mcap_source: str = "") -> float:
    """Convert market cap in local currency to USD.

    If mcap_source is 'fmp_profile', FMP already returns USD — skip conversion.
    Returns 0.0 if mcap_local is invalid.

    IMPORTANT - subdivision currencies:
    yfinance reports info["currency"] as the *price* currency (e.g. ZAc, ILA, GBp),
    but info["marketCap"] is ALWAYS in the base currency (ZAR, ILS, GBP respectively).
    We remap subdivision currencies to their base before the FX lookup so that the
    x0.01 subdivision_scale in _get_fx_rate_to_usd is never applied to marketCap.
    GBp is already handled by the GBp->GBP entry in _CURRENCY_TO_YAHOO_FX.
    """
    if not mcap_local or mcap_local <= 0:
        return 0.0
    if mcap_source == "fmp_profile" or (currency or "").upper() in ("USD", "USX", ""):
        return float(mcap_local)
    # Remap subdivision currencies: marketCap is in base currency, not subdivisions
    _MCAP_CURRENCY_REMAP = {"ZAC": "ZAR", "ZAc": "ZAR", "ILA": "ILS"}
    currency = _MCAP_CURRENCY_REMAP.get(currency, currency)
    return float(mcap_local) * _get_fx_rate_to_usd(currency)

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
# load_universe() imported from universe.py
# ────────────────────────────────────────────────────────────────

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
    """Method 4: derive EPS from net income / shares.
    Tries quarterly_income_stmt first; falls back to annual income_stmt.
    Last resort when all other methods fail. No estimate vs actual, but confirms earnings history.
    Annual fallback recovers ~236 tickers (JO, AX, L, PA, SW) where yfinance has no quarterly data.
    """
    def _rows_from_inc(inc, method_tag: str) -> List[Dict]:
        ni_label = next((l for l in ["Net Income", "NetIncome", "Net Income Common Stockholders"] if l in inc.index), None)
        if ni_label is None:
            return []
        ni_series = inc.loc[ni_label].dropna().sort_index()
        shares_label = next((l for l in ["Diluted Average Shares", "BasicAverageShares", "Ordinary Shares Number"] if l in inc.index), None)
        shares_series = inc.loc[shares_label].dropna().sort_index() if shares_label else None
        rows = []
        for date_col, ni in ni_series.items():
            eps_val = None
            if shares_series is not None and date_col in shares_series.index:
                sh = float(shares_series[date_col])
                if sh and sh > 0:
                    eps_val = round(float(ni) / sh, 4)
            # Also grab revenue if available
            rev_val = None
            for rev_label in ["Total Revenue", "TotalRevenue", "Revenue"]:
                if rev_label in inc.index and date_col in inc.loc[rev_label].index:
                    rv = inc.loc[rev_label][date_col]
                    if rv is not None and np.isfinite(float(rv)) and float(rv) > 0:
                        rev_val = float(rv)
                    break
            rows.append({
                "date": str(pd.Timestamp(date_col).date()),
                "eps_estimate": None,
                "eps_reported": eps_val,
                "eps_surprise_pct": None,
                "revenue_estimate": None,
                "revenue_reported": rev_val,
                "_method": method_tag,
            })
        return rows

    try:
        # Try quarterly first
        inc_q = tk.quarterly_income_stmt
        if inc_q is not None and not inc_q.empty:
            rows = _rows_from_inc(inc_q, "income_stmt_derived")
            if rows:
                return rows
        # Fall back to annual (recovers JO, AX, L, PA, SW etc)
        inc_a = tk.income_stmt
        if inc_a is not None and not inc_a.empty:
            rows = _rows_from_inc(inc_a, "income_stmt_annual_derived")
            if rows:
                return rows
        return []
    except Exception:
        return []


def _fetch_revenue_estimates_yahoo(tk, ticker: str) -> List[Dict]:
    """Fetch forward revenue + EPS estimates from Yahoo Finance quoteSummary endpoint.

    Uses yfinance's authenticated session (tk._data.get_raw_json) so the crumb/cookie
    handshake is handled automatically — unlike the previous bare requests.get() which
    failed silently for all tickers in production (GitHub Actions IPs blocked).

    Modules used:
      earningsTrend  → forward revenue + EPS estimates for next 2-4 quarters
      earningsHistory → past EPS actuals + estimates (revenue rarely present)

    Returns list of {date, revenue_estimate, eps_estimate} — FORWARD LOOKING.
    These are stored in out["forward_estimates"], NOT merged into past earnings_dates rows.
    earningsTrend dates are upcoming quarters; they will never match past eps_reported rows.

    Historical note: Yahoo removed revenueActual from earningsHistory. That field returns
    null universally. revenue_reported for past quarters comes from income_stmt (Phase A).
    """
    rows = []
    try:
        # Use yfinance's authenticated session — handles crumb/cookie automatically.
        # This is the correct approach; bare requests.get() is blocked on GitHub Actions.
        # ticker is the full Yahoo symbol (e.g. "AAPL", "ASML.AS", "7203.T")
        url = (
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            f"?modules=earningsTrend%2CearningsHistory&corsDomain=finance.yahoo.com"
        )
        raw = None
        # Primary: use yfinance internal authenticated session.
        # NOTE: do NOT pass handle_404=True — that kwarg does not exist in yfinance 1.2.0
        # and throws TypeError which is silently swallowed, returning [] for every ticker.
        # 404s (ticker genuinely missing from Yahoo) are handled by the except below.
        try:
            raw = tk._data.get_raw_json(url)
        except Exception:
            pass
        # Fallback: try via requests using yfinance cookie session if available
        if not raw:
            try:
                import requests as _req
                sess = getattr(tk._data, "session", None) or _req
                resp = sess.get(url, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
            except Exception:
                pass

        if not raw:
            return rows

        result = (raw.get("quoteSummary") or {}).get("result") or []
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

        # Build rows from earningsTrend (forward) separately from earningsHistory (past)
        # Tag each row with _is_forward so callers can distinguish future vs past quarters.
        # earningsTrend = FORWARD: has rev_estimate + eps_estimate for upcoming quarters
        # earningsHistory = PAST: has eps_actual + eps_estimate; rev_actual always null

        # Forward rows from earningsTrend
        for d, t_data in trend_by_qtr.items():
            rev_est = t_data.get("rev_estimate")
            eps_est = t_data.get("eps_estimate")
            if rev_est is None and eps_est is None:
                continue  # nothing useful in this trend item
            row = {
                "date": d + "-01",
                "revenue_estimate": rev_est,
                "revenue_reported": None,  # earningsHistory.revenueActual always null
                "eps_estimate": eps_est,
                "eps_reported": None,
                "_method": "yahoo_quotesummary",
                "_is_forward": True,   # upcoming quarter — store in forward_estimates
            }
            # Supplement with earningsHistory data for the same quarter if present
            h_data = hist_by_qtr.get(d, {})
            if row["eps_estimate"] is None:
                row["eps_estimate"] = h_data.get("eps_estimate")
            rows.append(row)

        # Past rows from earningsHistory (EPS actuals only — no revenue)
        for d, h_data in hist_by_qtr.items():
            if d in trend_by_qtr:
                continue  # already handled above as a forward row
            eps_act = h_data.get("eps_actual")
            eps_est = h_data.get("eps_estimate")
            if eps_act is None and eps_est is None:
                continue
            rows.append({
                "date": d + "-01",
                "revenue_estimate": h_data.get("rev_estimate"),  # always None in practice
                "revenue_reported": h_data.get("rev_actual"),    # always None in practice
                "eps_estimate": eps_est,
                "eps_reported": eps_act,
                "_method": "yahoo_quotesummary",
                "_is_forward": False,  # past quarter — do NOT include in forward_estimates
            })

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

        # ── Zombie / delisted auto-detection ────────────────────────────────
        # Tickers with no price AND no financials are dead — skip all 4 methods.
        try:
            last_price = getattr(tk.fast_info, "last_price", None)
            if last_price is None or last_price == 0:
                try:
                    _inc_check = tk.quarterly_income_stmt
                    no_fins = _inc_check is None or _inc_check.empty
                except Exception:
                    no_fins = True
                if no_fins:
                    return {
                        "ticker": ticker,
                        "inactive": True,
                        "inactive_reason": "no_price_no_financials",
                        "earnings_dates": [],
                        "quarterly_revenue": [],
                        "catalyst_events": [],
                        "info": {},
                        "error": "auto_inactive",
                    }
        except Exception:
            pass  # fast_info unavailable — proceed normally

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

        # ── Block 2c: Yahoo quoteSummary → forward_estimates ─────────────────
        # earningsTrend gives FORWARD-LOOKING estimates (next 2-4 quarters).
        # These are stored in out["forward_estimates"] — a separate list, NOT
        # merged into earnings_dates[] (past rows won't date-match upcoming quarters).
        #
        # forward_estimates is the seed for the estimates snapshot repository:
        # each daily run captures current consensus; over time this becomes historical.
        # When a quarter closes and actuals arrive, the snapshot shows what consensus
        # was at various points before the release — estimate revision history.
        #
        # Fix v0.6.5: switched from bare requests.get() (blocked on GitHub Actions —
        # 0 results in production) to tk._data.get_raw_json() which uses yfinance's
        # authenticated session (crumb/cookie handled automatically).
        try:
            yq_rows = _fetch_revenue_estimates_yahoo(tk, ticker)
            if yq_rows:
                # Store forward-looking rows only (_is_forward=True = from earningsTrend)
                # earningsHistory rows (_is_forward=False) are past quarters — exclude
                fwd = [r for r in yq_rows if r.get("_is_forward")
                       and (r.get("revenue_estimate") is not None or r.get("eps_estimate") is not None)]
                if fwd:
                    out["forward_estimates"] = fwd
                    out["_yahoo_qs_fwd_rows"] = len(fwd)
                    out["_yahoo_qs_fwd_rev"] = sum(1 for r in fwd if r.get("revenue_estimate") is not None)
                # Also enrich upcoming rows in best_eps with rev_estimate from earningsTrend
                # Only touch rows where eps_reported is None (upcoming quarters)
                # Use ±2 month tolerance — earningsTrend end-date vs announcement date differ
                yq_by_qtr: Dict[str, Dict] = {}
                for r in yq_rows:
                    if not r.get("_is_forward"):
                        continue  # skip earningsHistory past rows
                    d = (r.get("date") or "")[:7]
                    if d:
                        yq_by_qtr[d] = r
                for row in best_eps:
                    if row.get("eps_reported") is not None:
                        continue  # past row — skip (revenue_reported handled in Phase A step 3)
                    d = (row.get("date") or "")[:7]
                    candidates = [d]
                    try:
                        _y2, _m2 = int(d[:4]), int(d[5:7])
                        for _delta in [-1, 1, -2, 2]:
                            _nm2 = _m2 + _delta; _ny2 = _y2 + (_nm2-1)//12; _nm2 = ((_nm2-1)%12)+1
                            candidates.append(f"{_ny2:04d}-{_nm2:02d}")
                    except Exception:
                        pass
                    for _key2 in candidates:
                        if _key2 in yq_by_qtr:
                            qd = yq_by_qtr[_key2]
                            if row.get("revenue_estimate") is None and qd.get("revenue_estimate"):
                                row["revenue_estimate"] = qd["revenue_estimate"]
                                row["_rev_est_source"] = "yahoo_qs"
                            if row.get("eps_estimate") is None and qd.get("eps_estimate"):
                                row["eps_estimate"] = qd["eps_estimate"]
                                row["_eps_est_source"] = "yahoo_qs"
                            break
        except Exception as _yq_err:
            out["_yahoo_qs_error"] = str(_yq_err)

        # ── Block 2d: earningsHistory EPS → backfill past earnings_dates ─────
        # earningsHistory returns past 4 quarters with eps_estimate (analyst
        # consensus before release) and eps_reported (actual). These are now
        # available from _fetch_revenue_estimates_yahoo as _is_forward=False rows.
        # We merge their eps_estimate into past best_eps rows where it is missing.
        # This recovers non-US EPS estimate coverage: India/Korea/Taiwan/Japan had
        # 0% eps_estimate on past rows because earningsHistory was never used here.
        # Tagged _eps_est_source='yahoo_earnings_history' for attribution.
        try:
            if yq_rows:
                past_hist = [
                    r for r in yq_rows
                    if not r.get("_is_forward") and r.get("eps_estimate") is not None
                ]
                if past_hist:
                    hist_by_qtr: Dict[str, Dict] = {}
                    for r in past_hist:
                        d = (r.get("date") or "")[:7]
                        if d:
                            hist_by_qtr[d] = r
                    merged_hist = 0
                    for row in best_eps:
                        if row.get("eps_reported") is None:
                            continue  # forward row — handled in Block 2c
                        if row.get("eps_estimate") is not None:
                            continue  # already have an estimate from earlier source
                        d = (row.get("date") or "")[:7]
                        cands = [d]
                        try:
                            _yh, _mh = int(d[:4]), int(d[5:7])
                            for _dh in [-1, 1, -2, 2]:
                                _nmh = _mh + _dh
                                _nyh = _yh + (_nmh - 1) // 12
                                _nmh = ((_nmh - 1) % 12) + 1
                                cands.append(f"{_nyh:04d}-{_nmh:02d}")
                        except Exception:
                            pass
                        for _key_h in cands:
                            if _key_h in hist_by_qtr:
                                row["eps_estimate"] = hist_by_qtr[_key_h]["eps_estimate"]
                                row["_eps_est_source"] = "yahoo_earnings_history"
                                merged_hist += 1
                                break
                    if merged_hist:
                        out["_earnings_history_eps_merged"] = merged_hist
        except Exception:
            pass  # non-fatal
        out["earnings_dates"] = best_eps
        out["eps_method"] = best_method

        # ── Phase A step 3: Link quarterly_revenue into earnings_dates ──────────
        # Both datasets are now in memory from Phase A (Block 1 + Block 2).
        # This is a pure join — no network call, no API, entirely free.
        # Runs here so paid fallbacks (5b FMP, 5c Finnhub) only fire for
        # genuine gaps where yfinance has no income statement at all (~14%).
        #
        # Skip revenue_source="annual_estimated" rows (annual÷4 proxy) — those
        # are not real quarterly actuals and would corrupt beat/miss scoring.
        try:
            _qr_rows = out.get("quarterly_revenue", [])
            if _qr_rows:
                _qr_by_month: Dict[str, float] = {}
                for _qr in _qr_rows:
                    if _qr.get("revenue_source") == "annual_estimated":
                        continue
                    _d = (_qr.get("date") or "")[:7]
                    _rev = _qr.get("revenue")
                    if _d and _rev is not None and np.isfinite(float(_rev)) and float(_rev) > 0:
                        _qr_by_month[_d] = float(_rev)
                _filled_qr_a = 0
                for _row in out["earnings_dates"]:
                    if _row.get("revenue_reported") is not None:
                        continue
                    _d = (_row.get("date") or "")[:7]
                    _cands = [_d]
                    try:
                        _y, _m = int(_d[:4]), int(_d[5:7])
                        for _delta in [-1, -2, 1, 2]:
                            _nm = _m + _delta; _ny = _y + (_nm-1)//12; _nm = ((_nm-1)%12)+1
                            _cands.append(f"{_ny:04d}-{_nm:02d}")
                    except Exception:
                        pass
                    for _key in _cands:
                        if _key in _qr_by_month:
                            _row["revenue_reported"] = _qr_by_month[_key]
                            _row["_rev_act_source"] = "yf_income_stmt"
                            _filled_qr_a += 1
                            break
                if _filled_qr_a:
                    out["_qr_linkage_filled"] = _filled_qr_a
        except Exception:
            pass  # non-fatal — Phase B can still fill gaps
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

    # 5a) investing.com — DISABLED (v0.6.0).
    # GitHub Actions IPs are blocked by investing.com — confirmed 0 successful calls
    # in production (gc_state.json shows zero _investing_com_filled entries).
    # Leaving the enrich_estimates_investing_com() function intact for local dev use,
    # but removing the call from the hot path to eliminate per-ticker timeout overhead.
    # To re-enable locally: uncomment the block below.
    #
    # if _missing_estimates_count() > 0:
    #     try:
    #         filled_ic = enrich_estimates_investing_com(
    #             out["earnings_dates"], ticker,
    #             quarterly_revenue=out.get("quarterly_revenue"),
    #         )
    #         if filled_ic:
    #             out["_investing_com_filled"] = filled_ic
    #     except Exception as _ic_err:
    #         out["_investing_com_error"] = str(_ic_err)

    # 5b) FMP analyst-estimates via /stable/earnings endpoint.  [PAID — runs after yfinance]
    # Only called when _missing_estimates_count() > 0 — i.e. after 5d (yfinance income_stmt
    # linkage) has already filled revenue_reported for ~86% of universe for free.
    # FMP fills genuine gaps: tickers where quarterly_income_stmt was empty or didn't match.
    #
    # Endpoint is US-centric: only bare symbols work (no exchange suffix).
    # get_fmp_symbol(): ADR_MAP lookup first (2330.TW→TSM), then bare symbol fallback.
    # v0.6.1: reverted v0.6.0 _yahoo_to_fmp change — suffixed symbols (.AS/.DE) return nothing.
    fmp_key_enrich = os.environ.get("FMP_API_KEY", "").strip()
    if fmp_key_enrich and _missing_estimates_count() > 0 and out.get("earnings_dates"):
        try:
            from urllib.parse import urlencode
            import urllib.request as _ureq
            # get_fmp_symbol: ADR_MAP lookup first (2330.TW→TSM, 7203.T→TM etc.),
            # then bare symbol fallback for /stable/earnings US-centric endpoint.
            sym_fmp = get_fmp_symbol(ticker)
            est_data = None
            # /stable/earnings endpoint — available on Starter plan.
            # Returns: epsEstimated, revenueEstimated, epsActual, revenueActual per quarter.
            qs = urlencode({"symbol": sym_fmp, "limit": 16, "apikey": fmp_key_enrich})
            try:
                with _ureq.urlopen(f"{_FMP_BASE}/earnings?{qs}", timeout=8) as r:
                    raw = json.loads(r.read().decode())
                    if isinstance(raw, list) and raw:
                        est_data = raw
            except Exception:
                pass
            if est_data:
                fmp_est_by_qtr = {}
                for q in est_data:
                    d = str(q.get("date", ""))[:7]
                    # /stable/earnings field names
                    eps_avg = _safe_float(q.get("epsEstimated") or q.get("estimatedEpsAvg"))
                    rev_avg = _safe_float(q.get("revenueEstimated") or q.get("estimatedRevenueAvg"))
                    rev_act = _safe_float(q.get("revenueActual"))
                    if d:
                        fmp_est_by_qtr[d] = {"eps": eps_avg, "rev": rev_avg, "rev_act": rev_act}
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
                        # Use FMP revenue actual directly (more reliable than income stmt for beat/miss)
                        if (row.get("revenue_reported") is None
                                and q_est.get("rev_act") is not None
                                and np.isfinite(q_est["rev_act"]) and q_est["rev_act"] > 0):
                            row["revenue_reported"] = q_est["rev_act"]
                            row["_rev_act_source"] = "fmp"
                            filled_fmp += 1
                        elif row.get("revenue_reported") is None and key in rev_act_by_qtr:
                            row["revenue_reported"] = rev_act_by_qtr[key]
                            row["_rev_act_source"] = "fmp_income_stmt"
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
        from curl_cffi.requests import Session as CffiSession
    except ImportError:
        return 0

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return 0

    bare = ticker.split(".")[0].upper()   # NVDA, AAPL etc.
    impersonate = "chrome124"

    # ── Session warm-up ──────────────────────────────────────────────────────
    # investing.com sets session cookies on the homepage that subsequent requests
    # need to pass bot detection. Without cookies each request looks cold/robotic.
    # We reuse a module-level session so the cookie jar is shared across tickers.
    if not hasattr(enrich_estimates_investing_com, "_ic_session"):
        sess = CffiSession(impersonate=impersonate)
        try:
            warm = sess.get(
                "https://www.investing.com/",
                headers={
                    "Accept": "text/html",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=15,
            )
            # Small pause after warm-up before first real request
            time.sleep(2.0)
        except Exception:
            pass
        enrich_estimates_investing_com._ic_session = sess
        enrich_estimates_investing_com._slug_cache: Dict[str, str] = {}
    _sess = enrich_estimates_investing_com._ic_session
    _slug_cache: Dict[str, str] = enrich_estimates_investing_com._slug_cache

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

    # Known slugs for ~300 tickers — investing.com slugs are unpredictable
    # (companies rebrand, IC keeps old names). Dynamic discovery doesn't work
    # because their search API returns articles not quotes. This table covers
    # the full Nasdaq 100 + top S&P 500 constituents by market cap.
    SLUG_OVERRIDES: Dict[str, str] = {
        # ── Mega-cap tech ──────────────────────────────────────────────────
        "AAPL":  "apple-computer-inc",
        "MSFT":  "microsoft-corp",
        "NVDA":  "nvidia-corp",
        "GOOGL": "google-inc",
        "GOOG":  "google-inc-c",
        "META":  "facebook-inc",
        "AMZN":  "amazon-com-inc",
        "TSLA":  "tesla-motors",
        "AVGO":  "avago-technologies",
        # ── Financials ─────────────────────────────────────────────────────
        "JPM":   "jp-morgan-chase",
        "V":     "visa",
        "MA":    "mastercard",
        "BAC":   "bank-of-america",
        "WFC":   "wells-fargo",
        "GS":    "goldman-sachs-group",
        "MS":    "morgan-stanley",
        "AXP":   "american-express",
        "BLK":   "blackrock",
        "SCHW":  "charles-schwab",
        "C":     "citigroup",
        "USB":   "us-bancorp",
        "PNC":   "pnc-financial-services",
        "TFC":   "truist-financial",
        "COF":   "capital-one-financial",
        "BX":    "blackstone",
        "CB":    "chubb",
        "ICE":   "intercontinental-exchange",
        "CME":   "cme-group",
        "AON":   "aon-plc",
        "MMC":   "marsh-mclennan",
        "MCO":   "moodys-corp",
        "SPGI":  "sp-global",
        "FI":    "fiserv",
        "PYPL":  "paypal-holdings",
        "SYF":   "synchrony-financial",
        "DFS":   "discover-financial-services",
        # ── Healthcare ─────────────────────────────────────────────────────
        "JNJ":   "johnson-johnson",
        "UNH":   "unitedhealth",
        "LLY":   "eli-lilly",
        "ABT":   "abbott-laboratories",
        "MRK":   "merck---co",
        "TMO":   "thermo-fisher-scientific",
        "DHR":   "danaher",
        "ABBV":  "abbvie",
        "PFE":   "pfizer",
        "BMY":   "bristol-myers-squibb",
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
        "HCA":   "hca-holdings",
        "CI":    "cigna",
        "ELV":   "elevance-health",
        "CVS":   "cvs-health",
        "MCK":   "mckesson",
        "CAH":   "cardinal-health",
        "COR":   "cencora",
        "BSX":   "boston-scientific",
        "SYK":   "stryker",
        "ZBH":   "zimmer-biomet-holdings",
        "BDX":   "becton-dickinson",
        "BAX":   "baxter-intl",
        "HOLX":  "hologic",
        "IQV":   "iqvia-holdings",
        "A":     "agilent-technologies",
        "MTD":   "mettler-toledo",
        "WAT":   "waters-corp",
        "RMD":   "resmed",
        "PODD":  "insulet",
        # ── Energy ─────────────────────────────────────────────────────────
        "XOM":   "exxon-mobil",
        "CVX":   "chevron",
        "COP":   "conocophillips",
        "EOG":   "eog-resources",
        "SLB":   "schlumberger",
        "MPC":   "marathon-petroleum",
        "PSX":   "phillips-66",
        "VLO":   "valero-energy",
        "OXY":   "occidental-petroleum",
        "PXD":   "pioneer-natural-resources",
        "HES":   "hess",
        "DVN":   "devon-energy",
        "HAL":   "halliburton",
        "BKR":   "baker-hughes",
        "FANG":  "diamondback-energy",
        "TRGP":  "targa-resources",
        "WMB":   "williams-companies",
        "KMI":   "kinder-morgan",
        "OKE":   "oneok",
        # ── Consumer staples ───────────────────────────────────────────────
        "WMT":   "walmart",
        "PG":    "procter-gamble",
        "KO":    "coca-cola",
        "PM":    "philip-morris-intl",
        "MO":    "altria-group",
        "COST":  "costco-whsl-corp-new",
        "PEP":   "pepsico",
        "MDLZ":  "mondelez-international",
        "KHC":   "kraft-heinz",
        "MKC":   "mccormick",
        "GIS":   "general-mills",
        "K":     "kellogg",
        "CPB":   "campbell-soup",
        "HRL":   "hormel-foods",
        "SJM":   "j-m-smucker",
        "CLX":   "clorox",
        "EL":    "estee-lauder",
        "CL":    "colgate-palmolive",
        "CHD":   "church-dwight",
        "KMB":   "kimberly-clark",
        # ── Consumer discretionary ─────────────────────────────────────────
        "TSLA":  "tesla-motors",
        "HD":    "home-depot",
        "MCD":   "mcdonalds",
        "NKE":   "nike",
        "SBUX":  "starbucks-corp",
        "LOW":   "lowes-companies",
        "TGT":   "target",
        "NFLX":  "netflix,-inc.",
        "BKNG":  "priceline-com-inc",
        "ABNB":  "airbnb",
        "MAR":   "marriott-intl",
        "HLT":   "hilton-worldwide-holdings",
        "RCL":   "royal-caribbean",
        "CCL":   "carnival",
        "LVS":   "las-vegas-sands",
        "WYNN":  "wynn-resorts",
        "MGM":   "mgm-resorts-intl",
        "ORLY":  "oreilly-automotive",
        "AZO":   "autozone",
        "ROST":  "ross-stores-inc",
        "DLTR":  "dollar-tree-inc",
        "DG":    "dollar-general",
        "BBY":   "best-buy",
        "TJX":   "tjx",
        "MNST":  "monster-beverage",
        "KDP":   "keurig-dr-pepper",
        # ── Industrials ────────────────────────────────────────────────────
        "GE":    "general-electric",
        "CAT":   "caterpillar",
        "DE":    "deere",
        "HON":   "honeywell-intl",
        "RTX":   "raytheon-technologies",
        "LMT":   "lockheed-martin",
        "NOC":   "northrop-grumman",
        "GD":    "general-dynamics",
        "BA":    "boeing",
        "UPS":   "united-parcel",
        "FDX":   "fedex",
        "DAL":   "delta-air-lines",
        "UAL":   "united-airlines-holdings",
        "AAL":   "american-airlines",
        "LUV":   "southwest-airlines",
        "EMR":   "emerson-electric",
        "ETN":   "eaton",
        "IR":    "ingersoll-rand",
        "PH":    "parker-hannifin",
        "DOV":   "dover",
        "ITW":   "illinois-tool-works",
        "SWK":   "stanley-black-decker",
        "MMM":   "3m",
        "GWW":   "w-w-grainger",
        "FAST":  "fastenal-co",
        "PCAR":  "paccar-inc",
        "ODFL":  "old-dominion-freight",
        "CSX":   "csx",
        "NSC":   "norfolk-southern",
        "UNP":   "union-pacific",
        "XYL":   "xylem",
        "CARR":  "carrier-global",
        "OTIS":  "otis-worldwide",
        "ADP":   "automatic-data-processing",
        "CTAS":  "cintas-corp",
        "PAYX":  "paychex-inc",
        "CPRT":  "copart-inc",
        "VRSK":  "verisk-analytics",
        "AXON":  "axon-enterprise",
        # ── Technology ─────────────────────────────────────────────────────
        "AMD":   "adv-micro-device",
        "QCOM":  "qualcomm-inc",
        "AMAT":  "applied-matls-inc",
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
        "INTC":  "intel",
        "TXN":   "texas-instruments",
        "INTU":  "intuit",
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
        "TTD":   "the-trade-desk",
        "DASH":  "doordash",
        "COIN":  "coinbase-global",
        "PLTR":  "palantir-technologies",
        "APP":   "applovin",
        "RBLX":  "roblox",
        "ZM":    "zoom-video-communications",
        "MTCH":  "match-group",
        "NOW":   "servicenow",
        "CRM":   "salesforce",
        "ORCL":  "oracle",
        "SAP":   "sap",
        "ACN":   "accenture",
        "IBM":   "ibm",
        "HPQ":   "hewlett-packard",
        "HPE":   "hp-enterprise",
        "DELL":  "dell-technologies",
        "NTAP":  "network-appliance",
        "WDC":   "western-digital",
        "STX":   "seagate-technology",
        "CTSH":  "cognizant-technology-solutions",
        "IT":    "gartner",
        "EPAM":  "epam-systems",
        "GLOB":  "globant",
        "AKAM":  "akamai-technologies",
        "FFIV":  "f5-networks",
        "JNPR":  "juniper-networks",
        "KEYS":  "keysight-technologies",
        "TDY":   "teledyne-technologies",
        "TRMB":  "trimble",
        # ── Telecom / media ────────────────────────────────────────────────
        "CMCSA": "comcast-corp-new",
        "CHTR":  "charter-communications",
        "WBD":   "warner-bros-discovery",
        "SIRI":  "sirius-xm-holdings",
        "TTWO":  "take-two-interactive",
        "EA":    "electronic-arts",
        "ATVI":  "activision-blizzard",
        "T":     "at-t",
        "VZ":    "verizon",
        "TMUS":  "t-mobile-us",
        # ── Utilities / REIT ───────────────────────────────────────────────
        "LIN":   "linde-plc",
        "CEG":   "constellation-energy",
        "XEL":   "xcel-energy",
        "EXC":   "exelon-corp",
        "ENPH":  "enphase-energy",
        "NEE":   "nextera-energy",
        "DUK":   "duke-energy",
        "SO":    "southern",
        "D":     "dominion-energy",
        "AEP":   "american-electric-power",
        "PCG":   "pacific-gas-electric",
        "WEC":   "wec-energy-group",
        "ES":    "eversource-energy",
        "PEG":   "public-service-enterprise-group",
        "AWK":   "american-water-works",
        "AMT":   "american-tower",
        "CCI":   "crown-castle",
        "PLD":   "prologis",
        "EQIX":  "equinix",
        "DLR":   "digital-realty-trust",
        "PSA":   "public-storage",
        "EQR":   "equity-residential",
        "AVB":   "avalonbay-communities",
        "O":     "realty-income",
        "WELL":  "welltower",
        # ── Materials ──────────────────────────────────────────────────────
        "FCX":   "freeport-mcmoran",
        "NEM":   "newmont",
        "NUE":   "nucor",
        "STLD":  "steel-dynamics",
        "ALB":   "albemarle",
        "LYB":   "lyondellbasell-industries",
        "DOW":   "dow",
        "DD":    "dupont",
        "PPG":   "ppg-industries",
        "SHW":   "sherwin-williams",
        "ECL":   "ecolab",
        "APD":   "air-products-chemicals",
        "IFF":   "intl-flavors-fragrances",
        "EMN":   "eastman-chemical",
        "CE":    "celanese",
        # ── Foreign / ADR ──────────────────────────────────────────────────
        "ASML":  "asml-holding",
        "AZN":   "astrazeneca",
        "NVO":   "novo-nordisk",
        "MELI":  "mercadolibre",
        "PDD":   "pinduoduo",
        "NTES":  "netease",
        "ROP":   "roper-technologies",
        "TSM":   "taiwan-semiconductor",
        "SONY":  "sony",
        "TM":    "toyota-motor",
        "HMC":   "honda-motor",
        "RACE":  "ferrari",
        "SAP":   "sap",
        "SHOP":  "shopify",
        "SQ":    "block",
        "UBER":  "uber-technologies",
        "LYFT":  "lyft",
        "SNAP":  "snap",
        "PINS":  "pinterest",
        "TWTR":  "twitter",
        "SPOT":  "spotify-technology",
        # ── EV / clean energy ──────────────────────────────────────────────
        "RIVN":  "rivian-automotive",
        "LCID":  "lucid-group",
        "NIO":   "nio",
        "XPEV":  "xpeng",
        "LI":    "li-auto",
        # ── Additional Nasdaq 100 ──────────────────────────────────────────
        "HON":   "honeywell-intl",
        "CTAS":  "cintas-corp",
        "PAYX":  "paychex-inc",
        "PCAR":  "paccar-inc",
        "CPRT":  "copart-inc",
        "ODFL":  "old-dominion-freight",
        "VRSK":  "verisk-analytics",
        "MAR":   "marriott-intl",
        "IDXX":  "idexx-laboratories",
        "BIIB":  "biogen-idec-inc",
        "ILMN":  "illumina-inc",
        "DXCM":  "dexcom",
        "ALGN":  "align-technology",
        "GEHC":  "ge-healthcare",
        "MELI":  "mercadolibre",
        "NXPI":  "nxp-semiconductors",
        "MCHP":  "microchip-technology",
        "FTNT":  "fortinet-inc",
        "ADI":   "analog-devices",
        "FANG":  "diamondback-energy",
    }

    if bare in SLUG_OVERRIDES:
        slug = SLUG_OVERRIDES[bare]
    elif bare in _slug_cache:
        slug = _slug_cache[bare]
    else:
        # Not in slug table — try bare ticker lowercase as a best-effort guess.
        # Many tickers match their slug exactly (e.g. JPM → jpm-earnings works).
        # If the page 404s or has no earnings table it fails silently in step 2.
        slug = bare.lower()

    # Save discovered slug to cache for reuse across the run
    if slug:
        _slug_cache[bare] = slug

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

    # 2b) Revenue + EPS consensus from FMP /stable/earnings endpoint (Starter plan compatible).
    # Replaces the old /analyst-estimates call which returns 402 on Starter plan.
    try:
        import urllib.request as _ureq2
        from urllib.parse import urlencode as _ue2
        qs2 = _ue2({"symbol": sym, "limit": 16, "apikey": api_key})
        with _ureq2.urlopen(f"https://financialmodelingprep.com/stable/earnings?{qs2}", timeout=8) as _r2:
            earn_data = json.loads(_r2.read().decode())
        if earn_data and isinstance(earn_data, list):
            earn_by_qtr: Dict[str, Dict] = {}
            for q in earn_data:
                d = str(q.get("date", ""))[:7]
                if d:
                    earn_by_qtr[d] = {
                        "eps_est": _safe_float(q.get("epsEstimated")),
                        "rev_est": _safe_float(q.get("revenueEstimated")),
                        "rev_act": _safe_float(q.get("revenueActual")),
                    }
            filled = 0
            for row in out.get("earnings_dates", []):
                d = (row.get("date") or "")[:7]
                candidates = [d]
                try:
                    y, m = int(d[:4]), int(d[5:7])
                    for delta in [-1, -2, 1, 2]:
                        nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                        candidates.append(f"{ny:04d}-{nm:02d}")
                except Exception: pass
                for key in candidates:
                    if key not in earn_by_qtr:
                        continue
                    eq = earn_by_qtr[key]
                    if row.get("eps_estimate") is None and eq["eps_est"] is not None and np.isfinite(eq["eps_est"]):
                        row["eps_estimate"] = eq["eps_est"]
                        row["_eps_est_source"] = "fmp_earnings"
                        filled += 1
                    if row.get("revenue_estimate") is None and eq["rev_est"] is not None and np.isfinite(eq["rev_est"]) and eq["rev_est"] > 0:
                        row["revenue_estimate"] = eq["rev_est"]
                        row["_rev_est_source"] = "fmp_earnings"
                        filled += 1
                    if row.get("revenue_reported") is None and eq["rev_act"] is not None and np.isfinite(eq["rev_act"]) and eq["rev_act"] > 0:
                        row["revenue_reported"] = eq["rev_act"]
                        row["_rev_act_source"] = "fmp_earnings"
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
    mcap_skipped = 0
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

        # ── Dead-market skip (v0.6.0) ─────────────────────────────────────────
        # Exchanges KL (Malaysia), PS (Philippines), AD (UAE Abu Dhabi) return
        # 0 usable data from yfinance — skip entirely to save API calls.
        t_suffix = t.rsplit(".", 1)[-1] if "." in t else "US"
        if t_suffix in DEAD_MARKET_SUFFIXES:
            if t not in cache:
                cache[t] = {
                    "ticker": t, "inactive": True,
                    "inactive_since": now.isoformat(),
                    "inactive_reason": "dead_market_no_yfinance_support",
                    "quarterly_revenue": [], "earnings_dates": [],
                    "catalyst_events": [], "info": {}, "data_gap_alert": True,
                }
            results[t] = cache[t]
            dead_skipped += 1
            continue

        # ── Market-cap floor filter ─────────────────────────────────────────────
        # RULES (v0.8.0+):
        #   • Floor: $1B USD across ALL markets (MIN_MCAP_US_EU = MIN_MCAP_OTHER = 1B)
        #   • mcap from yfinance is in LOCAL currency → convert to USD via live FX rates
        #   • ONLY exclude when we KNOW the mcap AND it falls below floor.
        #     If mcap is 0 / missing, let the ticker through — do NOT exclude on ignorance.
        #   • FMP profile mcap is already in USD (skip FX conversion for that source)
        #   • below_min_mcap is RE-VALIDATED weekly: a company that grows past the floor
        #     re-enters the pipeline automatically on the next weekly re-check.
        if t in cache and not cache[t].get("new_constituent"):

            if cache[t].get("below_min_mcap"):
                # ── Weekly re-check for previously below-floor tickers ──────────
                # Without this, a company that grows from $0.8B → $1.5B stays
                # frozen as below_min_mcap forever because we never re-fetch.
                _last_check = cache[t].get("_mcap_recheck_date", "")
                _today_str  = now.date().isoformat()
                _days_since = 999
                if _last_check:
                    try:
                        _days_since = (now.date() - dt.date.fromisoformat(_last_check)).days
                    except Exception:
                        pass
                if _days_since >= 7 or force:
                    try:
                        import yfinance as _yf_mc
                        _fresh_info = _yf_mc.Ticker(t).info or {}
                        _fresh_mc   = _fresh_info.get("marketCap") or _fresh_info.get("market_cap")
                        _fresh_cur  = (_fresh_info.get("currency")
                                       or (cache[t].get("info") or {}).get("currency") or "USD")
                        cache[t]["_mcap_recheck_date"] = _today_str
                        if _fresh_mc:
                            _fresh_mc_usd = _mcap_to_usd(float(_fresh_mc), _fresh_cur)
                            if _fresh_mc_usd >= MIN_MCAP_US_EU:
                                # Graduated past floor — clear flag, re-enter full pipeline
                                print(f"[gc-data] {t}: mcap grew to ${_fresh_mc_usd/1e9:.1f}B "
                                      f"— clearing below_min_mcap, scheduling full re-fetch")
                                cache[t].pop("below_min_mcap", None)
                                cache[t].pop("mcap_threshold", None)
                                cache[t]["_last_known_mcap"] = float(_fresh_mc)
                                cache[t]["_mcap_usd"]        = _fresh_mc_usd
                                if not cache[t].get("info"):
                                    cache[t]["info"] = {}
                                cache[t]["info"]["market_cap"] = float(_fresh_mc)
                                cache[t]["info"]["currency"]   = _fresh_cur
                                # Fall through to normal cache/fetch logic below
                            else:
                                # Still below floor — refresh stored mcap value and skip
                                cache[t]["_last_known_mcap"]   = float(_fresh_mc)
                                cache[t]["_mcap_usd"]          = _fresh_mc_usd
                                cache[t]["mcap_threshold"]      = MIN_MCAP_US_EU
                                results[t] = cache[t]
                                mcap_skipped += 1
                                continue
                        else:
                            # yfinance returned no mcap — keep flag, skip
                            results[t] = cache[t]
                            mcap_skipped += 1
                            continue
                    except Exception:
                        # Re-check fetch failed — keep flag, try again next week
                        cache[t]["_mcap_recheck_date"] = _today_str
                        results[t] = cache[t]
                        mcap_skipped += 1
                        continue
                else:
                    # Not yet due for weekly re-check — skip as-is
                    results[t] = cache[t]
                    mcap_skipped += 1
                    continue

            # ── Normal mcap evaluation (not previously flagged below floor) ──
            cached_mc = (
                (cache[t].get("info") or {}).get("market_cap")
                or cache[t].get("_last_known_mcap")
                or 0
            )
            try:
                cached_mc = float(cached_mc)
            except (TypeError, ValueError):
                cached_mc = 0.0
            import math as _math
            if cached_mc > 0 and not _math.isnan(cached_mc):
                _currency     = (cache[t].get("info") or {}).get("currency", "USD") or "USD"
                _mcap_src     = cache[t].get("_mcap_source", "")
                cached_mc_usd = _mcap_to_usd(cached_mc, _currency, _mcap_src)
                if cached_mc_usd > 0 and cached_mc_usd < MIN_MCAP_US_EU:
                    # Below $1B USD floor — mark, record recheck date, skip
                    cache[t]["below_min_mcap"]     = True
                    cache[t]["mcap_threshold"]      = MIN_MCAP_US_EU
                    cache[t]["_mcap_usd"]           = cached_mc_usd
                    cache[t]["_mcap_recheck_date"]  = now.date().isoformat()
                    results[t] = cache[t]
                    mcap_skipped += 1
                    continue
                # Valid mcap above floor — store USD equivalent for export tools
                if cached_mc_usd > 0:
                    cache[t]["_mcap_usd"] = cached_mc_usd
            # cached_mc == 0 → unknown → let through (never exclude on ignorance)

        if t in cache:
            cached = cache[t]
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

    cached_count = len(results) - dead_skipped - mcap_skipped
    print(f"[gc-data] universe={len(tickers)}, cached={cached_count}, "
          f"dead_skipped={dead_skipped}, mcap_filtered={mcap_skipped}, to_fetch={len(to_fetch)}")

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

    # ── Helper: does a cache entry contain real data worth preserving? ────────
    def _cache_has_data(entry: dict) -> bool:
        if not entry:
            return False
        return (
            any(e.get("eps_reported") is not None for e in entry.get("earnings_dates", []))
            or len(entry.get("quarterly_revenue", [])) >= 4
        )

    # ── First pass: yfinance (serial) ─────────────────────────────
    # Serial sequential loop — parallel fetch (v0.6.8) was reverted because
    # 4 concurrent workers share one yfinance crumb/cookie session and Yahoo
    # invalidates it within ~2 minutes, causing 88%+ 401 failures on both
    # 2026-03-11 runs (4,513 and 4,522 of 5,152 tickers empty).
    # Serial with per-ticker pauses keeps the crumb alive for the full run.
    yf_failed: List[str] = []
    _cache_preserved: int = 0   # tickers saved from cache on failed refetch
    _abort_fetch: bool = False   # Fix 3: set True if Yahoo is degraded early

    print(f"[gc-data] first pass: {len(ordered_fetch)} tickers, 1 worker (serial)")
    for i, t in enumerate(ordered_fetch):
        if i > 0 and i % 100 == 0:
            print(f"[gc-data] progress: {i}/{len(ordered_fetch)} fetched")
            time.sleep(0.75)   # Hard pause every 100 — resets Yahoo rate-limit window

        # ── Fix 3: Early abort on yfinance degradation ───────────
        # After first 200 tickers, if >50% are empty, Yahoo's crumb has
        # almost certainly expired. Abort remaining fetches and pull from
        # cache to avoid overwriting good data with empty results.
        if i == 200 and not _abort_fetch:
            failed_so_far = len(yf_failed)
            if failed_so_far >= 100:
                print(
                    f"[gc-data] ⚠️  ABORT: {failed_so_far}/200 tickers empty on first 200 — "
                    f"yfinance crumb likely expired. Preserving existing cache for remaining tickers."
                )
                _abort_fetch = True

        if _abort_fetch:
            # Use cache if available to avoid data loss
            old = cache.get(t, {})
            if _cache_has_data(old):
                preserved = dict(old)
                preserved["_used_cached"] = True
                preserved["_cache_fallback_reason"] = "fetch_aborted_yf_degraded"
                preserved["_cache_fallback_run_date"] = now.isoformat()
                results[t] = preserved
                _cache_preserved += 1
            else:
                results[t] = {
                    "ticker": t,
                    "error": "fetch_aborted_yf_degraded",
                    "fetched_at": now.isoformat(),
                }
            continue

        try:
            data = fetch_earnings_data(t)
            # Auto-inactive: zombie detected
            if data.get("inactive"):
                cache[t] = {**data, "inactive_since": now.isoformat()}
                results[t] = data
                continue

            has_past_eps = any(e.get("eps_reported") is not None for e in data.get("earnings_dates", []))
            has_rev = len(data.get("quarterly_revenue", [])) >= 4
            has_info = data.get("info", {}).get("revenue_growth") is not None
            is_failed = not has_past_eps and not has_rev and not has_info and "error" not in data

            # ── Fix 2: Cache preservation on failed fetch ─────────
            # If this refetch returned empty AND the existing cache has real data,
            # keep the old data rather than overwriting with empty.
            if is_failed:
                old = cache.get(t, {})
                if _cache_has_data(old):
                    preserved = dict(old)
                    preserved["_used_cached"] = True
                    preserved["_cache_fallback_reason"] = "yf_empty_on_refetch"
                    preserved["_cache_fallback_run_date"] = now.isoformat()
                    results[t] = preserved
                    _cache_preserved += 1
                    # Don't add to yf_failed — cache covers it; skip FMP
                else:
                    results[t] = data
                    yf_failed.append(t)
            else:
                results[t] = data
                # Fresh good data — clear any stale flags from prior runs
                results[t].pop("_used_cached", None)
                results[t].pop("_cache_fallback_reason", None)
                results[t].pop("_cache_fallback_run_date", None)
                # Fix A: persist last known good mcap as top-level field
                # Survives degraded runs that wipe info{} block
                import math as _math_a
                _fresh_mc = results[t].get("info", {}).get("market_cap")
                if _fresh_mc and not _math_a.isnan(float(_fresh_mc)) and float(_fresh_mc) > 0:
                    results[t]["_last_known_mcap"] = float(_fresh_mc)
                    # Also store USD equivalent (for export_universe and mcap filter)
                    _cur = (results[t].get("info") or {}).get("currency", "USD") or "USD"
                    _usd = _mcap_to_usd(float(_fresh_mc), _cur)
                    if _usd > 0:
                        results[t]["_mcap_usd"] = _usd
                elif cache.get(t, {}).get("_last_known_mcap"):
                    results[t]["_last_known_mcap"] = cache[t]["_last_known_mcap"]
                    if cache[t].get("_mcap_usd"):
                        results[t]["_mcap_usd"] = cache[t]["_mcap_usd"]

        except Exception as e:
            results[t] = {"ticker": t, "error": str(e), "fetched_at": now.isoformat()}
            yf_failed.append(t)

        ex = _exch(t)
        pause = 0.22 if ex == "US" else 0.11
        if i % 5 == 4:
            time.sleep(pause)

    if _cache_preserved:
        print(f"[gc-data] cache preserved: {_cache_preserved} tickers kept from prior run (yf returned empty)")
    if _abort_fetch:
        print(f"[gc-data] ⚠️  fetch aborted early — {_cache_preserved} tickers served from cache, "
              f"{len(yf_failed)} with no cache coverage")

    # ── yfinance retry pass (serial, 5-second cooldown) ──────────
    # Second attempt before involving FMP — covers transient throttle hits.
    # Skip retry entirely if we aborted (crumb is dead; retry would also fail).
    if yf_failed and not _abort_fetch:
        print(f"[gc-data] yfinance retry: {len(yf_failed)} tickers empty on first pass")
        time.sleep(3.5)
        still_failed: List[str] = []
        for i, t in enumerate(yf_failed):
            if i > 0 and i % 30 == 0:
                time.sleep(1.5)
            try:
                data = fetch_earnings_data(t)
                has_past_eps = any(e.get("eps_reported") is not None for e in data.get("earnings_dates", []))
                has_rev = len(data.get("quarterly_revenue", [])) >= 4
                has_info = data.get("info", {}).get("revenue_growth") is not None
                is_failed = not has_past_eps and not has_rev and not has_info and "error" not in data

                # Fix 2 applies on retry too
                if is_failed:
                    old = cache.get(t, {})
                    if _cache_has_data(old):
                        preserved = dict(old)
                        preserved["_used_cached"] = True
                        preserved["_cache_fallback_reason"] = "yf_empty_on_retry"
                        preserved["_cache_fallback_run_date"] = now.isoformat()
                        results[t] = preserved
                        _cache_preserved += 1
                    else:
                        results[t] = data
                        still_failed.append(t)
                else:
                    results[t] = data
                    results[t].pop("_used_cached", None)
                    results[t].pop("_cache_fallback_reason", None)
                    results[t].pop("_cache_fallback_run_date", None)

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

    # ── EU/DM rev-missing FMP batch (v0.6.1) ─────────────────────
    # Targets non-US developed-market tickers that have EPS estimates from yfinance
    # but no revenue estimates (yfinance never returns rev estimates globally).
    # These tickers pass yfinance fine so they never enter yf_failed — this is a
    # dedicated pass using fetch_fmp_single() with BARE SYMBOL (strip exchange suffix).
    # Bare symbol works because many large EU/DM companies have US cross-listings that
    # FMP tracks (ASML.AS→ASML on NASDAQ, SAP.DE→SAP on NYSE, MC.PA→MC nowhere but
    # FMP may still carry global income-statement data under the local ticker base).
    # Exchanges included: major EU + Canada + Australia (high FMP coverage probability).
    # APAC local-only markets excluded (2330.TW→2330 = no FMP match expected).
    # FMP_ALPHA_BATCH_SUFFIXES imported from universe.py
    # Covers all exchanges with alpha bare symbols that FMP can match
    fmp_key_rev = os.environ.get("FMP_API_KEY", "").strip()
    if fmp_key_rev:
        rev_missing: List[str] = []
        for t, v in results.items():
            if v.get("inactive") or v.get("below_min_mcap"):
                continue
            suffix = t.rsplit(".", 1)[-1] if "." in t else "US"
            if suffix not in FMP_ALPHA_BATCH_SUFFIXES:
                continue
            ed = v.get("earnings_dates", [])
            has_eps_est = any(e.get("eps_estimate") is not None for e in ed)
            has_rev_est = any(e.get("revenue_estimate") is not None for e in ed)
            if has_eps_est and not has_rev_est:
                rev_missing.append(t)

        if rev_missing:
            print(f"[gc-data] FMP alpha-batch (rev-missing): {len(rev_missing)} tickers "
                  f"(suffixes: {sorted({t.rsplit('.',1)[-1] for t in rev_missing})})")
            rev_enriched = 0
            fmp_date_miss = 0  # tickers where FMP returned data but date-match failed (issue #6)
            for i, t in enumerate(rev_missing):
                if i > 0 and i % 50 == 0:
                    print(f"[gc-data] EU/DM rev-missing: {i}/{len(rev_missing)}")
                    time.sleep(1.0)
                try:
                    # Use bare symbol — /stable/earnings and FMP income-statement are
                    # US-centric; bare symbol matches the US cross-listing where it exists.
                    bare = t.split(".")[0].upper()
                    fdata = fetch_fmp_single(bare, fmp_key_rev)
                    if fdata and fdata.get("earnings_dates"):
                        existing = results.get(t, {})
                        ed_existing = existing.get("earnings_dates", [])
                        ed_fmp = fdata.get("earnings_dates", [])
                        # Only merge revenue estimates — preserve existing EPS and actuals
                        fmp_by_date: dict = {}
                        for row in ed_fmp:
                            d = (row.get("date") or "")[:7]
                            if d:
                                fmp_by_date[d] = row
                        filled = 0
                        for row in ed_existing:
                            d = (row.get("date") or "")[:7]
                            frow = fmp_by_date.get(d)
                            if not frow:
                                # Try ±1 month
                                try:
                                    y, m = int(d[:4]), int(d[5:7])
                                    for delta in [-1, 1, -2, 2]:
                                        nm = m + delta; ny = y + (nm-1)//12; nm = ((nm-1)%12)+1
                                        frow = fmp_by_date.get(f"{ny:04d}-{nm:02d}")
                                        if frow:
                                            break
                                except Exception:
                                    pass
                            if frow:
                                if row.get("revenue_estimate") is None and frow.get("revenue_estimate") is not None:
                                    row["revenue_estimate"] = frow["revenue_estimate"]
                                    row["_rev_est_source"] = "fmp_eu_batch"
                                    filled += 1
                                if row.get("revenue_reported") is None and frow.get("revenue_reported") is not None:
                                    row["revenue_reported"] = frow["revenue_reported"]
                                    row["_rev_act_source"] = "fmp_eu_batch"
                        if filled:
                            existing["_fmp_eu_batch_filled"] = filled
                            results[t] = existing
                            rev_enriched += 1
                        else:
                            fmp_date_miss += 1  # FMP had data but no matching earnings_dates row
                except Exception as _eu_fmp_e:
                    pass
                time.sleep(0.22)
            print(f"[gc-data] EU/DM rev-missing: enriched {rev_enriched}/{len(rev_missing)} tickers with FMP rev estimates")
            if fmp_date_miss:
                print(f"[gc-data] EU/DM rev-missing: {fmp_date_miss} tickers had FMP data but no "
                      f"matching earnings_dates row (date mismatch — phantom coverage in counters)")

    # ── Tag data gaps + auto-inactive for persistent zero-data tickers ──────
    # data_gap_alert = True means this ticker has NO usable earnings data.
    # Used by scan mode to flag when a technical signal cannot be confirmed
    # with Star 2/3 due to missing data (different from a genuine miss).
    #
    # Auto-inactive: if a ticker has returned zero data on 3+ consecutive daily runs,
    # it is very likely a ghost (unsupported exchange suffix, delisted, no Yahoo support).
    # Mark inactive so it is excluded from active counts and not fetched again.
    # Threshold = 3 runs to avoid false-positives from temporary yfinance failures.
    #
    # DEGRADED-RUN GUARD (v0.6.9): If this run was aborted early OR >40% of active
    # tickers are empty (indicating a global yfinance outage, not per-ticker ghosts),
    # freeze the _no_data_runs counter entirely for this run. This prevents mass
    # false-inactivation after 3 consecutive outage runs (as happened 2026-03-11/12).
    _active_fetched = [
        v for t, v in results.items()
        if not v.get("inactive") and not v.get("below_min_mcap")
    ]
    _empty_this_run = sum(
        1 for v in _active_fetched
        if not any(e.get("eps_reported") is not None for e in v.get("earnings_dates", []))
        and len(v.get("quarterly_revenue", [])) < 4
        and not v.get("info", {}).get("revenue_growth")
        and not v.get("_used_cached")
    )
    _run_degraded = _abort_fetch or (
        len(_active_fetched) > 200
        and _empty_this_run / len(_active_fetched) > 0.40
    )
    if _run_degraded:
        print(
            f"[gc-data] ⚠️  Degraded run detected "
            f"({_empty_this_run}/{len(_active_fetched)} active tickers empty) — "
            f"auto-inactive counter FROZEN this run to prevent false ghost-marking."
        )

    auto_inactived = 0
    for t, v in results.items():
        if v.get("inactive") or v.get("below_min_mcap"):
            continue
        has_rev = len(v.get("quarterly_revenue", [])) >= 4
        has_eps = any(e.get("eps_reported") is not None for e in v.get("earnings_dates", []))
        has_info = bool(v.get("info", {}).get("revenue_growth"))
        gap = not (has_rev or has_eps or has_info)
        v["data_gap_alert"] = gap
        if gap:
            if not _run_degraded:
                # Only increment counter on clean runs — outage runs don't count
                prev_gaps = (cache.get(t) or {}).get("_no_data_runs", 0)
                v["_no_data_runs"] = prev_gaps + 1
                if v["_no_data_runs"] >= 3:
                    v["inactive"] = True
                    v["inactive_since"] = now.isoformat()
                    v["inactive_reason"] = "persistent_no_data_3_runs"
                    auto_inactived += 1
        else:
            v.pop("_no_data_runs", None)  # reset counter on any successful data fetch
    if auto_inactived:
        print(f"[gc-data] auto-inactived {auto_inactived} persistent zero-data tickers")

    # ── Fix B: FMP company-profile mcap fallback ─────────────────
    # For tickers that went through a fresh fetch but still have no market cap
    # (yfinance returns empty info for many EM/APAC exchanges), hit the FMP
    # /stable/company-profile endpoint which reliably returns mktCap.
    # Only runs when FMP_API_KEY is set and exchange is in FMP_ALPHA_BATCH_SUFFIXES.
    # Stores result in _last_known_mcap and info.market_cap so the mcap floor
    # filter works correctly on the next run without needing another FMP call.
    _fmp_key_mcap = os.environ.get("FMP_API_KEY", "").strip()
    if _fmp_key_mcap:
        _mcap_candidates = [
            t for t in results
            if not results[t].get("inactive")
            and not results[t].get("below_min_mcap")
            and not results[t].get("_last_known_mcap")
            and not (results[t].get("info") or {}).get("market_cap")
            and (t.rsplit(".", 1)[-1] if "." in t else "US") in FMP_ALPHA_BATCH_SUFFIXES
        ]
        if _mcap_candidates:
            print(f"[gc-data] Fix B: FMP mcap fallback for {len(_mcap_candidates)} tickers missing market cap")
            _mcap_filled = 0
            for _i, _t in enumerate(_mcap_candidates):
                if _i > 0 and _i % 50 == 0:
                    print(f"[gc-data] Fix B: mcap progress {_i}/{len(_mcap_candidates)}")
                    time.sleep(0.5)
                try:
                    _bare = _t.split(".")[0].upper()
                    _profile = _fmp_get("/company-profile", {"symbol": _bare}, _fmp_key_mcap)
                    if _profile and isinstance(_profile, dict):
                        _mkt = _profile.get("mktCap") or _profile.get("marketCap")
                        if _mkt:
                            try:
                                _mkt_f = float(_mkt)
                                if _mkt_f > 0:
                                    results[_t]["_last_known_mcap"] = _mkt_f
                                    if not results[_t].get("info"):
                                        results[_t]["info"] = {}
                                    results[_t]["info"]["market_cap"] = _mkt_f
                                    results[_t]["_mcap_source"] = "fmp_profile"
                                    _mcap_filled += 1
                            except (TypeError, ValueError):
                                pass
                    time.sleep(0.12)
                except Exception:
                    pass
            print(f"[gc-data] Fix B: FMP mcap filled {_mcap_filled}/{len(_mcap_candidates)} tickers")

    # ── Summary ───────────────────────────────────────────────────
    active_results = {k: v for k, v in results.items()
                      if not v.get("inactive") and not v.get("below_min_mcap")}
    ok = sum(1 for v in active_results.values() if "error" not in v)
    rev_ok = sum(1 for v in active_results.values() if len(v.get("quarterly_revenue", [])) >= 4)
    eps_ok = sum(1 for v in active_results.values() if any(e.get("eps_reported") for e in v.get("earnings_dates", [])))
    catalyst_ok = sum(1 for v in active_results.values() if v.get("catalyst_events"))
    fmp_count = sum(1 for v in active_results.values() if v.get("data_source") == "fmp_fallback")
    fmp_enriched = sum(1 for v in active_results.values() if v.get("_fmp_rev_estimates_filled"))
    gap_count = sum(1 for v in active_results.values() if v.get("data_gap_alert"))
    print(
        f"[gc-data] done: {ok}/{len(active_results)} active tickers success | "
        f"(dead/market_skipped={dead_skipped} mcap_filtered={mcap_skipped}) | "
        f"revenue_data: {rev_ok} | eps_history: {eps_ok} | "
        f"fmp_fallback: {fmp_count} | fmp_enriched: {fmp_enriched} | "
        f"catalyst_events: {catalyst_ok} | data_gaps: {gap_count}"
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

    # Revenue result for most recent quarter (only meaningful when real estimate exists)
    # NOTE: must be placed after _revenue_beat_for_row is defined above.
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
def _build_coverage_html(earnings_cache: Dict[str, Any]) -> str:
    """
    Build an HTML string with three tables for the daily email:
      1. EPS + Revenue top-level coverage (source breakdown)
      2. Per-country GC data layer coverage
      3. Estimate gap summary

    Returns a self-contained HTML fragment (no <html>/<body> wrappers) that
    scan.py can embed directly into the email body.

    Stored in gc_state["report_html_coverage"] after every data run.
    """
    _STYLE = (
        "font-family:Arial,sans-serif;font-size:13px;border-collapse:collapse;width:100%;"
        "margin-bottom:18px;"
    )
    _TH = "style='background:#1a1a2e;color:#e0e0e0;padding:7px 10px;text-align:left;border:1px solid #444;'"
    _TH_R = "style='background:#1a1a2e;color:#e0e0e0;padding:7px 10px;text-align:right;border:1px solid #444;'"
    _TD = "style='padding:6px 10px;border:1px solid #ddd;text-align:left;'"
    _TD_R = "style='padding:6px 10px;border:1px solid #ddd;text-align:right;'"
    _TD_WARN = "style='padding:6px 10px;border:1px solid #ddd;text-align:right;color:#c0392b;font-weight:bold;'"
    _TR_ALT = "style='background:#f7f9fc;'"
    _TR_NORM = ""
    _TR_TOT = "style='background:#e8f0fe;font-weight:bold;'"

    N = max(len(earnings_cache), 1)

    def pct(n: int) -> str:
        return f"{n} <span style='color:#666;font-size:11px;'>({n * 100 // N}%)</span>"

    def pct_of(n: int, total: int) -> str:
        return f"{n} <span style='color:#666;font-size:11px;'>({n * 100 // max(total,1)}%)</span>"

    # ── Compute source tallies (same logic as print_data_summary) ────────────
    _SOURCES = ["yfinance", "investing_com", "fmp", "finnhub", "yoy_proxy", "none"]
    eps_est_src: Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_act_src: Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_est_src: Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_con_src: Dict[str, int] = {s: 0 for s in _SOURCES}
    eps_rep_count = 0

    for data in earnings_cache.values():
        past_ed = [r for r in data.get("earnings_dates", [])
                   if r.get("eps_reported") is not None]
        if past_ed:
            eps_rep_count += 1

        if any(r.get("eps_estimate") is not None for r in past_ed):
            srcs = [r.get("_eps_est_source", "yfinance") for r in past_ed
                    if r.get("eps_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            eps_est_src[dom if dom in eps_est_src else "yfinance"] += 1
        elif any(r.get("_eps_est_source") == "yoy_proxy" for r in past_ed):
            eps_est_src["yoy_proxy"] += 1
        else:
            eps_est_src["none"] += 1

        qr = [r for r in data.get("quarterly_revenue", []) if r.get("revenue") is not None]
        if qr:
            qr_src = data.get("data_source", "yfinance")
            rev_act_src[qr_src if qr_src in rev_act_src else "yfinance"] += 1
        else:
            rev_act_src["none"] += 1

        if any(r.get("revenue_estimate") is not None for r in past_ed):
            srcs = [r.get("_rev_est_source", "yfinance") for r in past_ed
                    if r.get("revenue_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            rev_est_src[dom if dom in rev_est_src else "yfinance"] += 1
        else:
            rev_est_src["none"] += 1

        if any(r.get("revenue_reported") is not None for r in past_ed):
            srcs = [r.get("_rev_act_source", "yfinance") for r in past_ed
                    if r.get("revenue_reported") is not None]
            dom = max(set(srcs), key=srcs.count)
            rev_con_src[dom if dom in rev_con_src else "yfinance"] += 1
        else:
            rev_con_src["none"] += 1

    def _s(d: Dict[str, int], k: str) -> str:
        v = d.get(k, 0)
        return pct(v) if v > 0 else "<span style='color:#aaa;'>–</span>"

    # ── Staleness banner (Fix 2/3 — v0.6.9) ─────────────────────
    # When yfinance degraded and cache was used, show a prominent warning
    # before the coverage tables so the operator knows data is not fresh.
    stale_entries = [
        (k, v) for k, v in earnings_cache.items()
        if v.get("_used_cached") and not v.get("inactive") and not v.get("below_min_mcap")
    ]
    stale_count = len(stale_entries)
    active_n = sum(1 for v in earnings_cache.values()
                   if not v.get("inactive") and not v.get("below_min_mcap"))
    stale_pct = stale_count * 100 // max(active_n, 1)

    staleness_html = ""
    if stale_count > 0:
        # Find oldest original fetched_at among stale entries
        oldest_date = "unknown"
        try:
            dates = [
                v.get("fetched_at", "")
                for _, v in stale_entries
                if v.get("fetched_at")
            ]
            if dates:
                oldest_date = min(dates)[:10]  # YYYY-MM-DD only
        except Exception:
            pass
        # Find the degradation run date (when cache fallback was triggered)
        degradation_date = "unknown"
        try:
            run_dates = [
                v.get("_cache_fallback_run_date", "")
                for _, v in stale_entries
                if v.get("_cache_fallback_run_date")
            ]
            if run_dates:
                degradation_date = max(run_dates)[:10]
        except Exception:
            pass
        # Reason breakdown
        reasons: Dict[str, int] = {}
        for _, v in stale_entries:
            r = v.get("_cache_fallback_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        reason_str = "; ".join(f"{r}: {c}" for r, c in sorted(reasons.items()))

        staleness_html = (
            f"<div style='background:#fff3cd;border:2px solid #e67e22;border-radius:4px;"
            f"padding:12px 16px;margin-bottom:14px;font-family:Arial,sans-serif;'>"
            f"<b style='color:#c0392b;font-size:14px;'>⚠️ Data Freshness Warning</b>"
            f"<table style='font-family:Arial;font-size:12px;margin-top:8px;border-collapse:collapse;width:100%;'>"
            f"<tr><td style='padding:3px 8px;font-weight:bold;color:#555;width:220px;'>Tickers using cached data</td>"
            f"<td style='padding:3px 8px;color:#c0392b;font-weight:bold;'>{stale_count} of {active_n} active ({stale_pct}%)</td></tr>"
            f"<tr><td style='padding:3px 8px;font-weight:bold;color:#555;'>Oldest cached data date</td>"
            f"<td style='padding:3px 8px;'>{oldest_date}</td></tr>"
            f"<tr><td style='padding:3px 8px;font-weight:bold;color:#555;'>Degradation detected on</td>"
            f"<td style='padding:3px 8px;'>{degradation_date}</td></tr>"
            f"<tr><td style='padding:3px 8px;font-weight:bold;color:#555;'>Fallback reasons</td>"
            f"<td style='padding:3px 8px;'>{reason_str}</td></tr>"
            f"</table>"
            f"<p style='font-size:11px;color:#666;margin:8px 0 0;'>"
            f"Coverage numbers below reflect cached data for stale tickers. "
            f"Run gc_engine --mode data without --force at night to restore fresh coverage.</p>"
            f"</div>"
        )

    # ── Table 1: EPS + Revenue source coverage ───────────────────────────────
    html = staleness_html + (
        f"<h3 style='font-family:Arial;font-size:14px;margin:16px 0 6px;color:#1a1a2e;'>"
        f"📊 GC Data Layer — Coverage Summary (v{GC_VERSION})</h3>"
        f"<table style='{_STYLE}'>"
        f"<thead><tr>"
        f"<th {_TH}>Metric</th>"
        f"<th {_TH_R}>yfinance</th>"
        f"<th {_TH_R}>investing.com</th>"
        f"<th {_TH_R}>FMP</th>"
        f"<th {_TH_R}>Finnhub</th>"
        f"<th {_TH_R}>YoY proxy</th>"
        f"<th {_TH_R}>none</th>"
        f"</tr></thead><tbody>"
        f"<tr {_TR_ALT}>"
        f"<td {_TD}><b>EPS Estimate</b></td>"
        f"<td {_TD_R}>{_s(eps_est_src,'yfinance')}</td>"
        f"<td {_TD_R}>{_s(eps_est_src,'investing_com')}</td>"
        f"<td {_TD_R}>{_s(eps_est_src,'fmp')}</td>"
        f"<td {_TD_R}>{_s(eps_est_src,'finnhub')}</td>"
        f"<td {_TD_R}>{_s(eps_est_src,'yoy_proxy')}</td>"
        f"<td {_TD_R}>{_s(eps_est_src,'none')}</td>"
        f"</tr>"
        f"<tr {_TR_NORM}>"
        f"<td {_TD}><b>EPS Reported</b> (actual)</td>"
        f"<td {_TD_R}>{pct(eps_rep_count)}</td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}>{pct(N - eps_rep_count)}</td>"
        f"</tr>"
        f"<tr {_TR_ALT}>"
        f"<td {_TD}><b>Revenue Actuals</b> (income stmt)</td>"
        f"<td {_TD_R}>{_s(rev_act_src,'yfinance')}</td>"
        f"<td {_TD_R}>{_s(rev_act_src,'investing_com')}</td>"
        f"<td {_TD_R}>{_s(rev_act_src,'fmp')}</td>"
        f"<td {_TD_R}>{_s(rev_act_src,'finnhub')}</td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}>{_s(rev_act_src,'none')}</td>"
        f"</tr>"
        f"<tr {_TR_NORM}>"
        f"<td {_TD}><b>Revenue Estimate</b> (consensus)</td>"
        f"<td {_TD_R}>{_s(rev_est_src,'yfinance')}</td>"
        f"<td {_TD_R}>{_s(rev_est_src,'investing_com')}</td>"
        f"<td {_TD_R}>{_s(rev_est_src,'fmp')}</td>"
        f"<td {_TD_R}>{_s(rev_est_src,'finnhub')}</td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}>{_s(rev_est_src,'none')}</td>"
        f"</tr>"
        f"<tr {_TR_ALT}>"
        f"<td {_TD}><b>Revenue Reported</b> (paired w/ estimate)</td>"
        f"<td {_TD_R}>{_s(rev_con_src,'yfinance')}</td>"
        f"<td {_TD_R}>{_s(rev_con_src,'investing_com')}</td>"
        f"<td {_TD_R}>{_s(rev_con_src,'fmp')}</td>"
        f"<td {_TD_R}>{_s(rev_con_src,'finnhub')}</td>"
        f"<td {_TD_R}><span style='color:#aaa;'>–</span></td>"
        f"<td {_TD_R}>{_s(rev_con_src,'none')}</td>"
        f"</tr>"
        f"</tbody></table>"
    )

    # ── Table 2: Per-country coverage ────────────────────────────────────────
    _EXCH_TO_COUNTRY: Dict[str, str] = {
        "US": "🇺🇸 United States",  "TO": "🇨🇦 Canada",         "L":  "🇬🇧 United Kingdom",
        "DE": "🇩🇪 Germany",        "PA": "🇫🇷 France",          "AS": "🇳🇱 Netherlands",
        "MI": "🇮🇹 Italy",          "MC": "🇪🇸 Spain",           "SW": "🇨🇭 Switzerland",
        "ST": "🇸🇪 Sweden",         "OL": "🇳🇴 Norway",          "HE": "🇫🇮 Finland",
        "CO": "🇩🇰 Denmark",        "AT": "🇬🇷 Greece",          "VI": "🇦🇹 Austria",
        "IR": "🇮🇪 Ireland",        "LS": "🇵🇹 Portugal",        "WA": "🇵🇱 Poland",
        "BD": "🇭🇺 Hungary",        "PR": "🇨🇿 Czech Republic",  "T":  "🇯🇵 Japan",
        "HK": "🇭🇰 Hong Kong",      "KS": "🇰🇷 South Korea",     "TW": "🇹🇼 Taiwan",
        "SI": "🇸🇬 Singapore",      "AX": "🇦🇺 Australia",       "NZ": "🇳🇿 New Zealand",
        "NS": "🇮🇳 India",          "BO": "🇮🇳 India (BSE)",     "SA": "🇧🇷 Brazil",
        "JO": "🇿🇦 South Africa",   "MX": "🇲🇽 Mexico",          "JK": "🇮🇩 Indonesia",
        "BK": "🇹🇭 Thailand",       "KL": "🇲🇾 Malaysia",        "IS": "🇹🇷 Turkey",
        "TA": "🇮🇱 Israel",         "SR": "🇸🇦 Saudi Arabia",    "AD": "🇦🇪 UAE-Abu Dhabi",
        "DU": "🇦🇪 UAE-Dubai",      "QA": "🇶🇦 Qatar",           "SS": "🇨🇳 China-SH",
        "SZ": "🇨🇳 China-SZ",       "CA": "🇪🇬 Egypt",           "SN": "🇨🇱 Chile",
        "CL": "🇨🇴 Colombia",       "BR": "🇧🇪 Belgium",
    }

    from collections import defaultdict as _dd
    _crows: Dict[str, list] = _dd(list)
    for t_key, data in earnings_cache.items():
        exch = t_key.rsplit(".", 1)[-1] if "." in t_key else "US"
        country = _EXCH_TO_COUNTRY.get(exch, f".{exch}")
        past = [r for r in data.get("earnings_dates", [])
                if r.get("eps_reported") is not None]
        _crows[country].append({
            "er": len(past) > 0,
            "ee": any(r.get("eps_estimate") is not None for r in past),
            "rr": any(r.get("revenue_reported") is not None for r in past),
            "re": any(r.get("revenue_estimate") is not None for r in past),
        })

    html += (
        f"<h3 style='font-family:Arial;font-size:14px;margin:16px 0 6px;color:#1a1a2e;'>"
        f"🌍 Per-Country Data Coverage</h3>"
        f"<table style='{_STYLE}'>"
        f"<thead><tr>"
        f"<th {_TH}>Country</th>"
        f"<th {_TH_R}>N</th>"
        f"<th {_TH_R}>EPS rep</th>"
        f"<th {_TH_R}>EPS est</th>"
        f"<th {_TH_R}>Rev rep</th>"
        f"<th {_TH_R}>Rev est</th>"
        f"<th {_TH_R}>All-4</th>"
        f"<th {_TH_R}>EPS est only</th>"
        f"<th {_TH_R}>Rev est only</th>"
        f"<th {_TH_R}>No est ⚠</th>"
        f"</tr></thead><tbody>"
    )

    _ctot = {"n": 0, "er": 0, "ee": 0, "rr": 0, "re": 0, "a4": 0, "blind": 0, "eps_o": 0, "rev_o": 0}
    alt = False
    for country, rows in sorted(_crows.items(), key=lambda x: (-len(x[1]), x[0])):
        cn = len(rows)
        er  = sum(1 for r in rows if r["er"])
        ee  = sum(1 for r in rows if r["ee"])
        rr  = sum(1 for r in rows if r["rr"])
        re  = sum(1 for r in rows if r["re"])
        a4  = sum(1 for r in rows if r["er"] and r["ee"] and r["rr"] and r["re"])
        blind    = sum(1 for r in rows if r["er"] and not r["ee"] and not r["re"])
        eps_only = sum(1 for r in rows if r["er"] and r["ee"] and not r["re"])
        rev_only = sum(1 for r in rows if r["er"] and r["re"] and not r["ee"])

        def cp(v: int) -> str:
            bar_w = v * 60 // max(cn, 1)
            color = "#27ae60" if v * 100 // cn >= 80 else ("#e67e22" if v * 100 // cn >= 40 else "#e74c3c")
            return (
                f"<div style='display:inline-block;width:{bar_w}px;height:8px;"
                f"background:{color};border-radius:3px;margin-right:4px;vertical-align:middle;'></div>"
                f"{v} <span style='color:#888;font-size:11px;'>({v*100//cn}%)</span>"
            )

        _blind_val = blind if blind else '<span style="color:#27ae60">✓</span>'
        blind_td = (
            f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right;"
            f"color:#c0392b;font-weight:bold;'>{blind}</td>"
            if blind > 5 else
            f"<td {_TD_R}>{_blind_val}</td>"
        )
        tr_style = _TR_ALT if alt else _TR_NORM
        alt = not alt
        html += (
            f"<tr {tr_style}>"
            f"<td {_TD}>{country}</td>"
            f"<td {_TD_R}>{cn}</td>"
            f"<td {_TD_R}>{cp(er)}</td>"
            f"<td {_TD_R}>{cp(ee)}</td>"
            f"<td {_TD_R}>{cp(rr)}</td>"
            f"<td {_TD_R}>{cp(re)}</td>"
            f"<td {_TD_R}>{a4}</td>"
            f"<td {_TD_R}>{eps_only if eps_only else '–'}</td>"
            f"<td {_TD_R}>{rev_only if rev_only else '–'}</td>"
            f"{blind_td}</tr>"
        )
        _ctot["n"] += cn; _ctot["er"] += er; _ctot["ee"] += ee
        _ctot["rr"] += rr; _ctot["re"] += re; _ctot["a4"] += a4
        _ctot["blind"] += blind; _ctot["eps_o"] += eps_only; _ctot["rev_o"] += rev_only

    tn = max(_ctot["n"], 1)
    html += (
        f"<tr {_TR_TOT}>"
        f"<td {_TD}>TOTAL</td>"
        f"<td {_TD_R}>{_ctot['n']}</td>"
        f"<td {_TD_R}>{_ctot['er']} ({_ctot['er']*100//tn}%)</td>"
        f"<td {_TD_R}>{_ctot['ee']} ({_ctot['ee']*100//tn}%)</td>"
        f"<td {_TD_R}>{_ctot['rr']} ({_ctot['rr']*100//tn}%)</td>"
        f"<td {_TD_R}>{_ctot['re']} ({_ctot['re']*100//tn}%)</td>"
        f"<td {_TD_R}>{_ctot['a4']}</td>"
        f"<td {_TD_R}>{_ctot['eps_o']}</td>"
        f"<td {_TD_R}>{_ctot['rev_o']}</td>"
        f"<td {_TD_R} style='color:#c0392b;font-weight:bold;'>{_ctot['blind']}</td>"
        f"</tr></tbody></table>"
    )

    # ── Table 3: Estimate gap summary ─────────────────────────────────────────
    _all_rows = [r for rows in _crows.values() for r in rows]
    gap_none  = sum(1 for r in _all_rows if r["er"] and not r["ee"] and not r["re"])
    gap_eps   = sum(1 for r in _all_rows if r["er"] and r["ee"] and not r["re"])
    gap_rev   = sum(1 for r in _all_rows if r["er"] and r["re"] and not r["ee"])

    html += (
        f"<h3 style='font-family:Arial;font-size:14px;margin:16px 0 6px;color:#1a1a2e;'>"
        f"⚠️ Estimate Coverage Gaps</h3>"
        f"<table style='{_STYLE}'>"
        f"<thead><tr>"
        f"<th {_TH}>Gap type</th>"
        f"<th {_TH_R}>Tickers</th>"
        f"<th {_TH}>Action needed</th>"
        f"</tr></thead><tbody>"
        f"<tr {_TR_ALT}>"
        f"<td {_TD}>No estimates at all (EPS + Rev both missing)</td>"
        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right;"
        f"color:#c0392b;font-weight:bold;'>{gap_none}</td>"
        f"<td {_TD}>IC / FMP non-US expansion — top targets: India (588), Saudi/Brazil (96), Turkey (72), Indonesia (69), Thailand (68)</td>"
        f"</tr>"
        f"<tr {_TR_NORM}>"
        f"<td {_TD}>EPS estimate only — Rev estimate missing</td>"
        f"<td {_TD_R}>{gap_eps}</td>"
        f"<td {_TD}>FMP /stable/earnings for non-US; IC for UK/EU</td>"
        f"</tr>"
        f"<tr {_TR_ALT}>"
        f"<td {_TD}>Rev estimate only — EPS estimate missing</td>"
        f"<td {_TD_R}>{gap_rev}</td>"
        f"<td {_TD}>Rare — check FMP symbol mapping</td>"
        f"</tr>"
        f"</tbody></table>"
        f"<p style='font-family:Arial;font-size:11px;color:#888;margin:4px 0 0;'>"
        f"Cascade: yfinance → investing.com → FMP → Finnhub. "
        f"Without consensus estimate, revenue beat/miss cannot be computed.</p>"
    )

    return html


def print_data_summary(
    earnings_cache: Dict[str, Dict[str, Any]],
    universe_df: pd.DataFrame,
) -> None:
    """Print a summary of the earnings data quality and top growth stocks."""
    total = len(earnings_cache)
    has_rev = sum(1 for v in earnings_cache.values() if len(v.get("quarterly_revenue", [])) >= 4)
    # Count only tickers with at least one PAST eps_reported row (not forward-only rows).
    # Previously counted len(earnings_dates) >= 1 which included forward estimates with
    # eps_reported=None — that overcounted by ~968 tickers (issue #4).
    has_eps = sum(
        1 for v in earnings_cache.values()
        if any(r.get("eps_reported") is not None for r in v.get("earnings_dates", []))
    )
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

    # ── Data coverage diagnostic (HTML tables — render correctly in email) ──────
    # NOTE: This section outputs raw HTML so email clients render proper tables.
    # scan.py should embed this output in an <html><body> email (which it does via
    # stdout capture). The state["report_html_coverage"] is the canonical version;
    # this stdout HTML is a fallback for pipelines that capture print output.
    #
    # Data counted:
    #   1. Revenue ACTUALS   — quarterly_income_stmt  → quarterly_revenue
    #   2. Revenue ESTIMATES — analyst consensus      → revenue_estimate in earnings_dates
    #   3. EPS ESTIMATES     — analyst consensus      → eps_estimate in earnings_dates

    _SOURCES = ["yfinance", "investing_com", "fmp", "finnhub", "yoy_proxy", "none"]
    eps_est_src:    Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_act_src:    Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_est_src:    Dict[str, int] = {s: 0 for s in _SOURCES}
    rev_con_src:    Dict[str, int] = {s: 0 for s in _SOURCES}

    ic_filled_total = fmp_enrich_total = finnhub_filled_total = qr_linkage_total = 0

    for t_key, data in earnings_cache.items():
        past_ed = [r for r in data.get("earnings_dates", [])
                   if r.get("eps_reported") is not None]

        if any(r.get("eps_estimate") is not None for r in past_ed):
            srcs = [r.get("_eps_est_source", "yfinance")
                    for r in past_ed if r.get("eps_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            eps_est_src[dom if dom in eps_est_src else "yfinance"] += 1
        elif any(r.get("_eps_est_source") == "yoy_proxy" for r in past_ed):
            eps_est_src["yoy_proxy"] += 1
        else:
            eps_est_src["none"] += 1

        qr = [r for r in data.get("quarterly_revenue", []) if r.get("revenue") is not None]
        if qr:
            qr_src = data.get("data_source", "yfinance")
            rev_act_src[qr_src if qr_src in rev_act_src else "yfinance"] += 1
        else:
            rev_act_src["none"] += 1

        if any(r.get("revenue_estimate") is not None for r in past_ed):
            srcs = [r.get("_rev_est_source", "yfinance")
                    for r in past_ed if r.get("revenue_estimate") is not None]
            dom = max(set(srcs), key=srcs.count)
            rev_est_src[dom if dom in rev_est_src else "yfinance"] += 1
        else:
            rev_est_src["none"] += 1

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
        qr_linkage_total     += data.get("_qr_linkage_filled", 0)

    N = max(len(earnings_cache), 1)

    def _pn(n: int) -> str:
        return f"{n} ({n * 100 // N}%)"

    def _blank(n: int) -> str:
        return _pn(n) if n > 0 else "–"

    # ── Emit HTML tables (renders in email, readable as raw text in logs) ──────
    eps_rep_count = sum(
        1 for d in earnings_cache.values()
        if any(r.get("eps_reported") is not None
               for r in d.get("earnings_dates", []))
    )

    _S = "font-family:Arial,sans-serif;font-size:13px;border-collapse:collapse;width:100%;margin-bottom:18px;"
    _TH = "style='background:#1a1a2e;color:#e0e0e0;padding:7px 10px;text-align:left;border:1px solid #444;'"
    _THR = "style='background:#1a1a2e;color:#e0e0e0;padding:7px 10px;text-align:right;border:1px solid #444;'"
    _TD = "style='padding:6px 10px;border:1px solid #ddd;text-align:left;'"
    _TDR = "style='padding:6px 10px;border:1px solid #ddd;text-align:right;'"
    _TDW = "style='padding:6px 10px;border:1px solid #ddd;text-align:right;color:#c0392b;font-weight:bold;'"
    _ALT = "style='background:#f7f9fc;'"
    _TOT = "style='background:#e8f0fe;font-weight:bold;'"
    _H3 = "style='font-family:Arial;font-size:14px;margin:16px 0 6px;color:#1a1a2e;'"

    def _sv(d: Dict[str, int], k: str) -> str:
        v = d.get(k, 0)
        return f"{v} ({v*100//N}%)" if v > 0 else "–"

    print(f"\n<h3 {_H3}>📊 GC Data Layer — Coverage Summary (v{GC_VERSION})</h3>")
    print(f"<table style='{_S}'><thead><tr>")
    print(f"<th {_TH}>Metric</th><th {_THR}>yfinance</th><th {_THR}>investing.com</th>"
          f"<th {_THR}>FMP</th><th {_THR}>Finnhub</th><th {_THR}>YoY proxy</th><th {_THR}>none</th>")
    print(f"</tr></thead><tbody>")

    rows_src = [
        ("EPS Estimate",            eps_est_src, True),
        ("EPS Reported (actual)",   None, False),
        ("Revenue Actuals (income stmt)", rev_act_src, False),
        ("Revenue Estimate (consensus)",  rev_est_src, False),
        ("Revenue Reported (paired w/ est)", rev_con_src, False),
    ]
    for i, (label, src, has_yoy) in enumerate(rows_src):
        tr = _ALT if i % 2 == 0 else ""
        if src is None:
            print(f"<tr {tr}><td {_TD}><b>{label}</b></td>"
                  f"<td {_TDR}>{_pn(eps_rep_count)}</td>"
                  f"<td {_TDR}>–</td><td {_TDR}>–</td><td {_TDR}>–</td>"
                  f"<td {_TDR}>–</td><td {_TDR}>{_pn(N - eps_rep_count)}</td></tr>")
        else:
            yoy_td = f"<td {_TDR}>{_sv(src,'yoy_proxy')}</td>" if has_yoy else f"<td {_TDR}>–</td>"
            print(f"<tr {tr}><td {_TD}><b>{label}</b></td>"
                  f"<td {_TDR}>{_sv(src,'yfinance')}</td>"
                  f"<td {_TDR}>{_sv(src,'investing_com')}</td>"
                  f"<td {_TDR}>{_sv(src,'fmp')}</td>"
                  f"<td {_TDR}>{_sv(src,'finnhub')}</td>"
                  f"{yoy_td}"
                  f"<td {_TDR}>{_sv(src,'none')}</td></tr>")
    print(f"</tbody></table>")

    # ── Table 2: Per-country coverage (all countries, with gap breakdown) ──────
    _EXCH_TO_COUNTRY_FLAGS: Dict[str, str] = {
        "US": "🇺🇸 United States",  "TO": "🇨🇦 Canada",         "L":  "🇬🇧 United Kingdom",
        "DE": "🇩🇪 Germany",        "PA": "🇫🇷 France",          "AS": "🇳🇱 Netherlands",
        "MI": "🇮🇹 Italy",          "MC": "🇪🇸 Spain",           "SW": "🇨🇭 Switzerland",
        "ST": "🇸🇪 Sweden",         "OL": "🇳🇴 Norway",          "HE": "🇫🇮 Finland",
        "CO": "🇩🇰 Denmark",        "AT": "🇬🇷 Greece",          "VI": "🇦🇹 Austria",
        "IR": "🇮🇪 Ireland",        "LS": "🇵🇹 Portugal",        "WA": "🇵🇱 Poland",
        "BD": "🇭🇺 Hungary",        "PR": "🇨🇿 Czech Republic",  "T":  "🇯🇵 Japan",
        "HK": "🇭🇰 Hong Kong",      "KS": "🇰🇷 South Korea",     "TW": "🇹🇼 Taiwan",
        "SI": "🇸🇬 Singapore",      "AX": "🇦🇺 Australia",       "NZ": "🇳🇿 New Zealand",
        "NS": "🇮🇳 India",          "BO": "🇮🇳 India (BSE)",     "SA": "🇧🇷 Brazil",
        "JO": "🇿🇦 South Africa",   "MX": "🇲🇽 Mexico",          "JK": "🇮🇩 Indonesia",
        "BK": "🇹🇭 Thailand",       "KL": "🇲🇾 Malaysia",        "IS": "🇹🇷 Turkey",
        "TA": "🇮🇱 Israel",         "SR": "🇸🇦 Saudi Arabia",    "AD": "🇦🇪 UAE-Abu Dhabi",
        "DU": "🇦🇪 UAE-Dubai",      "QA": "🇶🇦 Qatar",           "SS": "🇨🇳 China-SH",
        "SZ": "🇨🇳 China-SZ",       "CA": "🇪🇬 Egypt",           "SN": "🇨🇱 Chile",
        "CL": "🇨🇴 Colombia",       "BR": "🇧🇪 Belgium",         "BD": "🇧🇩 Bangladesh",
        "MX": "🇲🇽 Mexico",         "WA": "🇵🇱 Poland",
    }
    from collections import defaultdict as _dd
    _country_rows: Dict[str, list] = _dd(list)
    for t_key, data in earnings_cache.items():
        exch = t_key.rsplit(".", 1)[-1] if "." in t_key else "US"
        country = _EXCH_TO_COUNTRY_FLAGS.get(exch, f".{exch}")
        past = [r for r in data.get("earnings_dates", [])
                if r.get("eps_reported") is not None]
        _country_rows[country].append({
            "n": len(data.get("earnings_dates", [])) + len(data.get("quarterly_revenue", [])),
            "has_eps_rep": len(past) > 0,
            "has_eps_est": any(r.get("eps_estimate") is not None for r in past),
            "has_rev_rep": any(r.get("revenue_reported") is not None for r in past),
            "has_rev_est": any(r.get("revenue_estimate") is not None for r in past),
        })

    print(f"<h3 {_H3}>🌍 Per-Country Data Coverage</h3>")
    print(f"<table style='{_S}'><thead><tr>")
    print(f"<th {_TH}>Country</th><th {_THR}>N</th>"
          f"<th {_THR}>EPS rep</th><th {_THR}>EPS est</th>"
          f"<th {_THR}>Rev rep</th><th {_THR}>Rev est</th>"
          f"<th {_THR}>All-4</th>"
          f"<th {_THR}>EPS est only</th>"
          f"<th {_THR}>Rev est only</th>"
          f"<th {_THR}>No est ⚠</th>")
    print(f"</tr></thead><tbody>")

    _ctot = {"n": 0, "er": 0, "ee": 0, "rr": 0, "re": 0, "a4": 0, "blind": 0, "eps_o": 0, "rev_o": 0}
    alt = False
    for country, rows in sorted(_country_rows.items(), key=lambda x: (-len(x[1]), x[0])):
        cn = len(rows)
        er  = sum(1 for r in rows if r["has_eps_rep"])
        ee  = sum(1 for r in rows if r["has_eps_est"])
        rr  = sum(1 for r in rows if r["has_rev_rep"])
        re  = sum(1 for r in rows if r["has_rev_est"])
        a4  = sum(1 for r in rows if r["has_eps_rep"] and r["has_eps_est"] and r["has_rev_rep"] and r["has_rev_est"])
        blind   = sum(1 for r in rows if r["has_eps_rep"] and not r["has_eps_est"] and not r["has_rev_est"])
        eps_only = sum(1 for r in rows if r["has_eps_rep"] and r["has_eps_est"] and not r["has_rev_est"])
        rev_only = sum(1 for r in rows if r["has_eps_rep"] and r["has_rev_est"] and not r["has_eps_est"])

        def _cp(v: int) -> str:
            p = v * 100 // max(cn, 1)
            c = "#27ae60" if p >= 80 else ("#e67e22" if p >= 40 else "#e74c3c")
            w = v * 50 // max(cn, 1)
            return (f"<div style='display:inline-block;width:{w}px;height:7px;"
                    f"background:{c};border-radius:2px;margin-right:4px;vertical-align:middle;'></div>"
                    f"{v} <span style='color:#888;font-size:11px;'>({p}%)</span>")

        blind_td_style = _TDW if blind > 5 else _TDR
        _blind_cell = blind if blind else '<span style="color:#27ae60">✓</span>'
        tr_style = _ALT if alt else ""
        alt = not alt
        print(f"<tr {tr_style}>"
              f"<td {_TD}>{country}</td>"
              f"<td {_TDR}>{cn}</td>"
              f"<td {_TDR}>{_cp(er)}</td>"
              f"<td {_TDR}>{_cp(ee)}</td>"
              f"<td {_TDR}>{_cp(rr)}</td>"
              f"<td {_TDR}>{_cp(re)}</td>"
              f"<td {_TDR}>{a4}</td>"
              f"<td {_TDR}>{eps_only if eps_only else '–'}</td>"
              f"<td {_TDR}>{rev_only if rev_only else '–'}</td>"
              f"<td {blind_td_style}>{_blind_cell}</td>"
              f"</tr>")
        _ctot["n"] += cn; _ctot["er"] += er; _ctot["ee"] += ee
        _ctot["rr"] += rr; _ctot["re"] += re; _ctot["a4"] += a4
        _ctot["blind"] += blind; _ctot["eps_o"] += eps_only; _ctot["rev_o"] += rev_only

    tn = max(_ctot["n"], 1)
    print(f"<tr {_TOT}>"
          f"<td {_TD}>TOTAL</td>"
          f"<td {_TDR}>{_ctot['n']}</td>"
          f"<td {_TDR}>{_ctot['er']} ({_ctot['er']*100//tn}%)</td>"
          f"<td {_TDR}>{_ctot['ee']} ({_ctot['ee']*100//tn}%)</td>"
          f"<td {_TDR}>{_ctot['rr']} ({_ctot['rr']*100//tn}%)</td>"
          f"<td {_TDR}>{_ctot['re']} ({_ctot['re']*100//tn}%)</td>"
          f"<td {_TDR}>{_ctot['a4']}</td>"
          f"<td {_TDR}>{_ctot['eps_o']}</td>"
          f"<td {_TDR}>{_ctot['rev_o']}</td>"
          f"<td {_TDW}>{_ctot['blind']}</td>"
          f"</tr></tbody></table>")

    # ── Table 3: Estimate gap analysis ──────────────────────────────────────────
    _all_rows_flat = [rw for rows in _country_rows.values() for rw in rows]
    _gap_none    = sum(1 for r in _all_rows_flat if r["has_eps_rep"] and not r["has_eps_est"] and not r["has_rev_est"])
    _gap_eps_o   = sum(1 for r in _all_rows_flat if r["has_eps_rep"] and r["has_eps_est"] and not r["has_rev_est"])
    _gap_rev_o   = sum(1 for r in _all_rows_flat if r["has_eps_rep"] and r["has_rev_est"] and not r["has_eps_est"])

    print(f"<h3 {_H3}>⚠️ Estimate Coverage Gaps</h3>")
    print(f"<table style='{_S}'><thead><tr>"
          f"<th {_TH}>Gap type</th><th {_THR}>Tickers</th><th {_TH}>Action needed</th>"
          f"</tr></thead><tbody>")
    print(f"<tr {_ALT}><td {_TD}>No estimates at all (EPS + Rev both missing)</td>"
          f"<td {_TDW}>{_gap_none}</td>"
          f"<td {_TD}>IC / FMP non-US expansion — top targets: India (588), Saudi/Brazil (96), Turkey (72), Indonesia (69), Thailand (68)</td></tr>")
    print(f"<tr><td {_TD}>EPS estimate only — Rev estimate missing</td>"
          f"<td {_TDR}>{_gap_eps_o}</td>"
          f"<td {_TD}>FMP /stable/earnings for non-US; IC for UK/EU</td></tr>")
    print(f"<tr {_ALT}><td {_TD}>Rev estimate only — EPS estimate missing</td>"
          f"<td {_TDR}>{_gap_rev_o}</td>"
          f"<td {_TD}>Rare — check FMP symbol mapping</td></tr>")
    print(f"</tbody></table>")
    print(f"<p style='font-family:Arial;font-size:11px;color:#888;margin:4px 0 12px;'>"
          f"Cascade: yfinance → investing.com → FMP → Finnhub. "
          f"Without consensus estimate, revenue beat/miss cannot be computed. "
          f"IC test run: 0 fields filled (not yet in state — run IC enrichment first).</p>")

    if ic_filled_total or fmp_enrich_total or finnhub_filled_total or qr_linkage_total:
        print(f"<p style='font-family:Arial;font-size:11px;color:#555;margin:4px 0;'>"
              f"<b>Fields enriched this run:</b> "
              f"investing.com: {ic_filled_total} | "
              f"FMP: {fmp_enrich_total} | "
              f"Finnhub: {finnhub_filled_total} | "
              f"QR linkage: {qr_linkage_total}</p>")

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
    has_any_eps_consensus = any(
        r.get("eps_estimate") is not None
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

    # No-coverage fallback: ticker has NO analyst estimates at all (no EPS, no revenue
    # consensus from any source). Qualify via 2 consecutive quarters of >=20% revenue
    # YoY growth. Clearly flagged so report shows this is growth-only, not a beat signal.
    # Requires actual quarterly revenue data (not info_fallback / annual_estimated).
    rev_yoy_recent  = rev_analytics.get("latest_yoy_growth")
    rev_yoy_prev    = rev_analytics.get("prev_yoy_growth")
    rev_src         = rev_analytics.get("revenue_source", "none")
    has_quarterly_rev = rev_src not in ("info_fallback", "annual_estimated", "none")
    meets_rev_growth_only = (
        not meets_dual_beat
        and not massive_catalyst
        and not data_gap_single
        and not meets_rev_only
        and not has_any_eps_consensus          # truly no analyst coverage at all
        and not has_any_rev_consensus
        and has_quarterly_rev
        and rev_yoy_recent is not None
        and rev_yoy_prev is not None
        and rev_yoy_recent >= 20.0             # latest Q >= 20% YoY
        and rev_yoy_prev >= 20.0              # prior Q >= 20% YoY (2 consecutive)
    )

    star2 = meets_dual_beat or massive_catalyst or data_gap_single or meets_rev_only or meets_rev_growth_only
    if out["data_gap_alert"]:
        out["star2_blocked"] = "no_data"
    elif star2:
        out["stars"] = 2
        if meets_dual_beat:
            out["star2_via"] = "dual_beat"
        elif massive_catalyst:
            out["star2_via"] = "catalyst"
        elif data_gap_single:
            out["star2_via"] = "data_gap_eps_only"   # EPS beat confirmed, rev estimates unavailable
        elif meets_rev_only:
            out["star2_via"] = "data_gap_rev_only"   # rev beat confirmed, EPS data weak
        else:
            out["star2_via"] = "rev_growth_only"     # no analyst coverage — 2Q ≥20% YoY growth

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

        # Build and store HTML coverage tables for the daily email
        # scan.py reads state["report_html_coverage"] and embeds it in the email body.
        state["report_html_coverage"] = _build_coverage_html(earnings_cache)
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
