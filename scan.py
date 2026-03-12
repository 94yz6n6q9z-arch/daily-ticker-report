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
- Added VP runway metric for VALIDATED signals: distance to nearest opposing HVN (%).
- v83: Fix HS/IHS neckline angle measurement by normalizing slope using median ATR on the reaction segment (not ATR(head)).
- v83: Ensure watchlist tickers display company names (NAME_OVERRIDES / yfinance fallback) instead of bare tickers in Section 4.
- v84: Fix floating neckline bug — store neckline through actual reaction points (T1→T2 / R1→R2) instead of (T1→last_bar).
- v84: Refresh neckline y-values from Close during state carry-forward to prevent stale values after price adjustments.
- v84: Extend neckline visually to chart right edge (solid T1→T2 + dotted projection).
- v84: Add RKT to FOCUS_TICKERS for full HS geometry deep-dive with chart in Section 4.
- v84: Remove broken "HS/IHS diagnosis (selected)" (referenced undefined `data` variable — never worked).
- v84: Remove "Watchlist big movers diagnostics" and "IHS early deep-dive" sections from Section 4.
- v84: Replace with "Focus tickers deep-dive" driven by FOCUS_TICKERS (NU, CEG, RKT) — always shown with full diagnostics + charts.
- v84: Add NAME_OVERRIDES for ASML, HOOD, OKLO.
- v84: Fix macro charts (VIX/EUR) appearing at bottom of email — stop stripping inline images (stripping caused email clients to render them as bottom-attached files).
- v84: Focus tickers: 0669.HK, TOST, TSN — add validation gate history table showing per-bar price/CLV/volume gate status, lifecycle labels, and exit reason.
- v84: Add download failure diagnostic to Signal engine health: breakdown by exchange suffix and country to identify OHLCV coverage gaps.
- v85: Focus tickers: 0669.HK, TOST, TSN with validation gate history (per-bar 3-gate table + lifecycle labels + exit reason).
- v85: Download diagnostic in Signal engine health: failure breakdown by exchange and country.
- v85: NAME_OVERRIDES: ASML, HOOD, OKLO. Remove Watchlist big movers / IHS early deep-dive sections. Fix macro charts in email.
- v86: Fix MSCI mapping script: reorder exchange suffix rules so Euronext/Nordic match before US catch-all. Adds NORDIC to Stockholm regex. Fixes 114 tickers across France, Sweden, Belgium, Finland, Portugal.
- v87: Fix yfinance rate-limiting: add 1.5s sleep between chunk downloads, 0.3s between individual retries, and a full second retry pass for still-missing tickers. Closes ~330 ticker coverage gap (Canada, UK, Australia, Germany, etc.).
- v88: Rewrite yf_download_chunk — group tickers by exchange suffix before chunking. Mixed-exchange bulk downloads cause Yahoo to silently drop non-US tickers. Each exchange group (.L, .TO, .DE, etc.) now downloads in its own homogeneous chunks. Removes all sleep/throttling (was counterproductive). Per-exchange progress logging.
- v89: ROOT CAUSE FIX — _clean_ticker regex [A-Z]+[.][A-Z]+ was converting exchange suffixes (.L, .DE, .TO, .AX, etc.) to hyphens (BP.L→BP-L). This destroyed ~400 international tickers. Now checks against known exchange suffixes before converting. Also: yfinance Ticker fallback for after-hours movers, diagnostic logging for yahoo_quote endpoint.
- v90: Fix Nordic share classes in MSCI script: "VOLV B"→"VOLV-B.ST" not "VOLVB.ST" (31 Swedish + 7 Danish tickers). Rewrite after-hours movers: primary source now yfinance Ticker.info (authenticated) instead of deprecated v7/finance/quote endpoint (was returning 403). Fixes AVGO and other after-hours missing movers.
- v91: Expand universe to MSCI World + MSCI EM (~2,641 tickers, ~1,400 net new). Add MSCI_EM_CLASSIFICATION_CSV path constant. Universe builder merges EM CSV if present, gracefully skips if not yet generated. Sector/company/country resolvers now cover EM tickers. update_msci_world_classification.py gains --universe world|em|both, SOURCE_CANDIDATES_EM (EIMI + EEM), 30+ new EM exchange suffix rules (.KS .TW .NS .BO .SA .JO .MX .SS .SZ .JK .BK .KL .SR .IS .WA etc.), and KNOWN_TICKER_OVERRIDES applied post-guessing. gc_engine.py: MSCI_EM_CSV path, load_universe() merges both CSVs, TICKER_OVERRIDES auto-corrects 7 bad mappings, compute_revenue_analytics() falls back to info.revenue_growth for markets with no quarterly data (recovers ~348 tickers: Japan, UK, Australia, France, Switzerland). gc-data.yml: refreshes both World + EM CSVs before data layer, adds lxml to install step, commits all 5 generated files.
- v92: Fix job timeout caused by bare EM numeric tickers (e.g. 6488, 3227 — missing .T/.NS/.BO suffix) flooding yfinance with 404 requests. Add _is_ghost_ticker() guard in universe builder: filters Bloomberg placeholders (pure numeric, *D suffix, bare dash, multi-dot) before they enter tech_scan_universe. Expand _clean_ticker _EXCHANGE_SUFFIXES to cover all EM markets (.TW .SS .SZ .JO .MX .JK .BK .KL .SR .AD .DU .WA .AT .CA .PS .QA .KQ etc.) so no EM suffix is ever mangled to a hyphen. Rename user-facing "MSCI World" labels to "MSCI World + EM" in Section 4C header and engine-health log. update_msci_world_classification.py and gc_engine.py gain version-tracker constants + changelog log (matching scan.py pattern).
- v93: Add DUMMY to _GHOST_RE. GC diagnostic: add revenue_beat_streak counter, fix star counts in engine health panel to show Star 2/3 separately.
- v94: Section 7 Growth Compounders Three-Star Signals. build_gc_three_star_section_md() reads ignition_signals from gc_state.json, renders summary table + OpenAI per-ticker brief (what company does + moat explanation). GC diagnostic shows Star 2 dual-beat vs Star 3 moat-confirmed counts.
- v95: Fix Star 2/3 broken counts. Revenue beat cascade: yfinance → investing.com scrape → FMP analyst-estimates → Finnhub → YoY proxy. Star 2 data-gap fallback: allow single beat (EPS or Rev) when all sources exhausted. Diagnostic shows per-source coverage counts.
- v96: Add FINNHUB_API_KEY env note. investing.com layer added between yfinance and FMP. (1) revenue_beat_streak now falls back to quarterly_revenue YoY>0 when yfinance lacks consensus revenue estimates — fixes NVDA/LLY always showing 0. (2) Add run_gc_ignition_scoring() called from main() after Star 1 — scores Star 2/3 for all Star 1 tickers, writes ignition_signals to gc_state.json. Section 7 now always rendered (shows placeholder when no 3-star signals). (3) Turkish .IS tickers normalised from TICKER-E.IS to TICKER.IS at load time in both scan.py and gc_engine.py.
- v97: Fix GC coverage tables broken in email (pipe-table rows had trailing \\n inside md.append() — "\n".join(md) then added a second newline between every row, which the markdown `tables` extension treats as paragraph breaks, rendering raw | text | instead of HTML tables). Fix: stripped trailing \\n from all table-row md.append() calls. Added full per-country coverage table (all ~40 markets, not just totals) with columns: N | EPS rep | EPS est | Rev rep | Rev est | All-4 | EPS est only | Rev est only | No est. Added estimate gap analysis table (no-est / eps-only / rev-only with top offending countries). Added Korea CSV (config/msci_korea_classification.csv) to the World/EM/Korea universe count line in engine health. Companion updates: gc_engine.py 0.5.3 (same HTML table fixes in _build_coverage_html and print_data_summary), update_msci_world_classification.py 1.6.0 (adds 6 new country ETFs: EWJ Japan, EWT Taiwan, MCHI China, EWH Hong Kong, KSA Saudi Arabia, EWM Malaysia — all produce clean Yahoo Finance numeric suffixes that the EM/World UCITS exports lose to the ghost-ticker filter).
- v98: Companion to gc_engine.py 0.6.2–0.6.3. Added ADR_MAP + get_fmp_symbol() to universe.py (25 APAC numeric→US ADR mappings: 2330.TW→TSM, 7203.T→TM etc.). FMP alpha-batch expanded from EU-only to all alpha-suffix exchanges globally (FMP_ALPHA_BATCH_SUFFIXES: +India, Brazil, Mexico, Turkey, Indonesia, S.Africa, Chile, Qatar, Kuwait, Singapore). Adds ~900 more tickers to the FMP revenue-missing batch. All exchange classification constants (DEAD_MARKET_SUFFIXES, EU_SUFFIXES, MIN_MCAP_US_EU, MIN_MCAP_OTHER, mcap_threshold, FMP_ALPHA_BATCH_SUFFIXES, ADR_MAP, get_fmp_symbol) now imported exclusively from universe.py — scan.py no longer defines any exchange logic itself.
- v99: Data source breakdown table rewritten — correct attribution for all 4 fields. Corrected wrong claim that APAC numeric markets (KS/TW/T/SS/SZ) have no EPS estimates from yfinance: earnings_dates HTML provides analyst consensus for ALL global markets. Revenue estimate is the only field where APAC numeric = 0%. Table now 5 columns (yf earn_dates | yf income_stmt | yf fwd_est | FMP | missing), dropping the defunct yf_quoteSummary and Finnhub columns. Added fwd_covered counter in table footer (earningsTrend tickers with forward_estimates, v0.6.5+). New diagnostic table: Forward revenue estimate coverage — per market, for every ticker with a yfinance EPS estimate, shows whether forward rev estimate came from yf earningsTrend (free), FMP only (paid), or is absent. This makes FMP residual value visible per run. Companion: gc_engine.py 0.6.5–0.6.6 (earningsTrend auth fix, Phase A linkage, _is_forward tagging, ±2 month tolerance, auto-inactive for persistent zero-data tickers). universe.py 1.2.0: load_universe() now filters DEAD_MARKET_SUFFIXES and KNOWN_DEAD_TICKERS at source; .DU added to DEAD_MARKET_SUFFIXES (95% dead confirmed). Effect: ~145 fewer tickers enter pipeline; active count projects below 5,000.
- v100: Option A universe filter. OHLCV download list now built from gc_state.json active set rather than raw load_universe() output. Tickers gc_engine has marked inactive or below_min_mcap are excluded from the yf_download_chunk call — eliminates OHLCV fetches for dead markets, below-mcap stocks, and persistent zero-data tickers. New MSCI constituents not yet seen by gc_engine pass through unfiltered for their first run (one free pass), then gc_engine trims them on the next cycle. gc_state absent (first run) falls back to full MSCI universe. Establishes explicit run-order dependency: gc_engine (mode=data) must complete and write gc_state.json BEFORE scan.py runs. Companion: universe.py 1.2.0, gc_engine.py 0.6.6.
- v101: Universe simplification companion. Removed MSCI_KR_CSV import (deleted from universe.py 1.4.0 — Korea now fully covered by msci_em_classification.csv via XLS-based pipeline). Removed korea_count from report stats block. Companion: universe.py 1.4.0, update_msci_world_classification.py 3.0.0.
- v102: Ticker normalization + exchange coverage fixes. Companion: universe.py 1.5.0, gc_engine.py 0.8.0.
  (1) _clean_ticker _EXCHANGE_SUFFIXES: added .AE (UAE unified), .TWO (Taiwan Gretai) — prevents mangling to hyphens.
  (2) _infer_country_from_ticker suffix_map: expanded from 18 → 40+ entries (added .KQ, .TWO, .AE, .IS, .SA, .NS, .BO, .BK, .JK, .SN, .WA, .AT, .VI, .IR, .LS, .MX, .JO, .SR, .QA, .KW, .NZ, .TW, .SI, .AD, .DU, .HE, .OL, .ST, .CO, .SW).
  (3) load_msci_csv_sectors(): replaced Turkish-only -E.IS normalization with full universe._normalize_ticker — now covers all markets (UAE, Thailand NVDR, London double-dot, class-share dot→dash).
  (4) Option A filter: comment clarifies below_min_mcap only excludes when gc_engine confirmed it — unknown mcap tickers pass through (rule: exclude only when known).
  (5) Coverage table _EXCH_FLAGS: added .AE (UAE unified), .KQ (KOSDAQ), .TWO (Taiwan Gretai), .BO (India BSE)."""
from __future__ import annotations
import argparse
import datetime as dt
from zoneinfo import ZoneInfo
import json
import math
import os
import re
import time
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

from universe import (
    load_universe,
    is_ghost_ticker,
    KNOWN_DEAD_TICKERS,
    CONFIG_DIR,
    DOCS_DIR,
    GC_STATE_PATH,
    MSCI_CSV,
    MSCI_EM_CSV,
    # Exchange classification
    DEAD_MARKET_SUFFIXES,
    EU_SUFFIXES,
    MIN_MCAP_US_EU,
    MIN_MCAP_OTHER,
    mcap_threshold,
    FMP_ALPHA_BATCH_SUFFIXES,
)

SCAN_VERSION: str = "v101"
# ----------------------------
# Public asset URLs (email-safe) + cache busting
# ----------------------------
PUBLIC_BASE_URL: str = ""   # e.g., https://<owner>.github.io/<repo>/
CACHE_BUST: str = ""        # e.g., YYYYMMDDHHMMSS
def _derive_public_base_url() -> str:
    """Best-effort public base URL for images so they render in email clients.
    We prefer GitHub Pages (docs/ published at repo root), because relative img/ paths
    do not resolve in email bodies.
    """
    # Explicit override
    u = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if u:
        return u.rstrip("/") + "/"
    # GitHub Actions provides owner/repo
    repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()  # e.g. owner/name
    if repo and "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner}.github.io/{name}/"
    # Fallback to your known repo (safe default)
    return "https://94yz6n6q9z-arch.github.io/daily-ticker-report/"
def _asset_url(src: str) -> str:
    """Turn a local path like 'img/foo.png' into an absolute URL + cache-bust.
    - Absolute URLs are returned unchanged.
    - Relative URLs are prefixed with PUBLIC_BASE_URL.
    - A cache-bust query param is appended when CACHE_BUST is set.
    """
    try:
        s = str(src or "").strip()
    except Exception:
        return src
    if not s:
        return s
    # Absolute URL? leave it.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        return s
    base = (PUBLIC_BASE_URL or "").strip()
    if base:
        s = base.rstrip("/") + "/" + s.lstrip("/")
    if CACHE_BUST:
        sep = "&" if "?" in s else "?"
        s = f"{s}{sep}v={CACHE_BUST}"
    return s
# Watchlist performance table (implemented locally)
# ----------------------------
# Default watchlist (user-defined)
# ----------------------------
# Ensures your full  watchlist is ALWAYS included when MODE=custom.
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
# Commodities (Yahoo Finance continuous futures symbols)
COMMODITY_TICKERS: List[str] = ["GC=F", "SI=F", "KC=F", "CC=F"]
COMMODITY_NAME_OVERRIDES: Dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "KC=F": "Coffee",
    "CC=F": "Cocoa",
}
# Force these tickers to always appear with charts + gate diagnosis in Section 4 (even if no live signal)
FOCUS_TICKERS = ["0669.HK", "TOST", "TSN"]
# Display name overrides (Section 6 + readability). Values should be FULL CAPS.
NAME_OVERRIDES = {
    "PLTR": "PALANTIR TECHNOLOGIES",
    "RRTL.DE": "RTL GROUP",
    "BYDDY": "BYD",
    "ANF": "ABERCROMBIE & FITCH",
    "MUV2.DE": "MUNICH RE",
    "NVO": "NOVO NORDISK",
    "CCJ": "CAMECO CORPORATION",
    "LEU": "CENTRUS ENERGY",
    "NAT": "NORDIC AMERICAN TANKERS",
    "FRO": "FRONTLINE PLC",
    "MAU.PA": "MAUREL & PROM S.A.",
    "INSW": "INTERNATIONAL SEAWAYS",
    "REP.MC": "REPSOL",
    "PSX": "PHILLIPS 66",
    "QBTS": "D-WAVE QUANTUM INC.",
    "TSM": "TSMC",
    "NFLX": "NETFLIX",
    "GOOGL": "GOOGLE",
    "UCG": "UNI CREDIT S.P.A.",
    "UCG.MI": "UNI CREDIT S.P.A.",
    "ARR": "ARMOUR RESIDENTIAL REIT",
    "NU": "NU HOLDINGS LTD.",
    "SMR": "NUSCALE POWER CORP.",
    "TNK": "TEEKAY TANKERS LTD.",
    "ASML": "ASML HOLDING N.V.",
    "HOOD": "ROBINHOOD MARKETS INC",
    "OKLO": "OKLO INC",
    "GC=F": "GOLD",
    "SI=F": "SILVER",
    "KC=F": "COFFEE",
    "CC=F": "COCOA",
}
WATCHLIST_GROUPS: Dict[str, List[str]] = {
    # EDA merged into this bucket
    "AI compute & semis (incl. EDA)": ["NVDA","ARM","AVGO","TSM","000660.KS","ASML","AMAT","LRCX","SNPS","CDNS"],
    "AI software/data": ["PLTR"],
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
    "Commodities": COMMODITY_TICKERS,
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
    # AI software/data
    "PLTR": "AI SW/Data",
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
    """Human-friendly labels for tickers in the report.
    - Exchange suffixes are removed for equities (e.g., MC.PA -> MC).
    - Commodity continuous futures (e.g., SI=F) are shown as the commodity name.
    """
    t = str(t).strip()
    # Commodities: prefer explicit commodity names over the Yahoo symbol.
    if t in COMMODITY_NAME_OVERRIDES:
        return COMMODITY_NAME_OVERRIDES[t]
    if t in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[t]
    base = _base_ticker(t)
    if base in COMMODITY_NAME_OVERRIDES:
        return COMMODITY_NAME_OVERRIDES[base]
    return DISPLAY_NAME_OVERRIDES.get(base, base)
# Ticker display labels: include segment tag when available, but hide exchange suffix.
TICKER_LABELS: Dict[str, str] = {t: f"{_display_name(t)} ({seg})" for t, seg in SEGMENT_TAGS.items()}
def display_ticker(t: str) -> str:
    """Plain display for tickers in tables/headers (no segment tags)."""
    return _display_name(t)
def display_ticker_tagged(t: str) -> str:
    """Optional: ticker with segment tag, e.g., NVDA (Compute/IP)."""
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
# CONFIG_DIR, DOCS_DIR, GC_STATE_PATH imported from universe.py
IMG_DIR = DOCS_DIR / "img"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DOCS_DIR / "state.json"
REPORT_PATH = DOCS_DIR / "report.md"
INDEX_PATH = DOCS_DIR / "index.md"
EMAIL_REPORT_PATH = DOCS_DIR / "email.md"
EMAIL_MD_PATH = DOCS_DIR / "email.md"
EMAIL_TXT_PATH = DOCS_DIR / "email.txt"
CUSTOM_TICKERS_PATH = CONFIG_DIR / "tickers_custom.txt"
SP500_LOCAL = CONFIG_DIR / "universe_sp500.txt"
NDX_LOCAL = CONFIG_DIR / "universe_nasdaq100.txt"
# MSCI CSV paths imported from universe.py (MSCI_CSV, MSCI_EM_CSV)
MSCI_WORLD_CLASSIFICATION_CSV = MSCI_CSV      # alias for backward compat
MSCI_EM_CLASSIFICATION_CSV = MSCI_EM_CSV      # alias for backward compat
# ----------------------------
# Config knobs
# ----------------------------
MOVER_THRESHOLD_PCT = 4.0
ATR_N = 14
ATR_CONFIRM_MULT = 0.5     # confirmed breakout/breakdown threshold
EARLY_MULT = 0.5           # early callout threshold (within 0.5 ATR)
VALIDATE_BARS = 3         # validated lifecycle window (bars): day0..day2 (all3, price+vol, all3)
DCB_EARLY_MAX_BARS = 5     # dead-cat-bounce EARLY expires after 5 bars from event low (fresh shock only)
DCB_EARLY_MAX_FROM_BOUNCE = 4  # ...and max 4 bars from bounce high
VOL_CONFIRM_MULT = 1.25   # volume must be >= 1.25x AvgVol(20) for CONFIRMED
CLV_BREAKOUT_MIN = 0.70   # CLV in [-1..+1] must be >= +0.70 for breakout confirmation
CLV_BREAKDOWN_MAX = -0.70  # CLV in [-1..+1] must be <= -0.70 for breakdown confirmation
LOOKBACK_DAYS = 190
# HS/IHS minimum formation duration (daily bars) to avoid too-short (≈2-3 week) false positives
HS_MIN_BARS = 30
HS_MIN_SIDE_BARS = 10
# HS/IHS maximum formation duration (daily bars) to avoid stale multi-month patterns
HS_MAX_BARS = 90
# Geometry diagnostics / guardrails
HS_SYMMETRY_MIN_RATIO = 0.70   # min(min(dL,dR)/max(dL,dR))
HS_VALLEY_ATR_MULT = 2.0       # valley depth threshold vs ATR at head
HS_SHOULDER_VALLEY_ATR_MULT = 1.0  # min shoulder-to-trough depth in ATR(head) units

# Neckline slope angle guardrail (degrees from horizontal).
# We measure the neckline angle as the raw slope between reaction points (T1→T2 for HS_TOP, R1→R2 for IHS), i.e., arctan(Δprice/Δbars).
# Neckline slope angle is capped symmetrically: abs(angle) ≤ 17.5° for both HS_TOP and IHS.
HS_NECKLINE_MAX_ANGLE_DEG = 17.5
HS_LOCAL_WINDOW = 3           # local max/min window for shoulder check (±3 bars)
# Maximum allowed lag between pattern completion (RS) and breakout/breakdown confirmation run start
HS_MAX_BREAKOUT_LAG_BARS = 30
HS_GEOM_CARRY_BARS = 30  # persist HS/IHS geometry up to 30 bars to survive pivot re-picks on big bars
BAND_GEOM_CARRY_BARS = 30  # persist band geometry (rect/tri/broaden) up to 30 bars since last validating touch
# Lifecycle: CONFIRMED is only day 0..1 of a new confirmed run. Day 2 becomes VALIDATED if the validation window holds.
CONFIRMED_MAX_AGE_BARS = 1
VALIDATED_MIN_AGE_BARS = 2
# Keep VALIDATED ongoing for at most this many bars after the breakout day (unless you change it).
VALIDATED_MAX_AGE_BARS = 30
# Exit rules (applied to CONFIRMED + VALIDATED signals for ALL formations)
EXIT_ENABLE = True
EXIT_GIVEBACK_ATR = 2.0  # exit when giveback >= 2 ATR_ref from peak excursion
# Dead Cat Bounce: event must be an overnight gap-down of at least 10% (open vs prior close)
DCB_MIN_GAP_PCT = 0.10
# Chart window (timeline) for all signal charts
CHART_WINDOW_DAYS = 190   # ~6 months
CHART_MIN_BARS = 120
# EARLY callouts must be fresh: pattern completion must be recent (prevents old formations resurfacing)
EARLY_MAX_AGE_FROM_PATTERN_END_BARS = 30
DOWNLOAD_PERIOD = "3y"
DOWNLOAD_INTERVAL = "1d"
CHUNK_SIZE = 80
MAX_CHARTS_EARLY = 30
MAX_CHARTS_CONFIRMED = 9999
MAX_CHARTS_VALIDATED = 9999
MAX_CHARTS_TRIGGERED = 18
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
# Volume Profile (VP) runway gate — deterministic, daily OHLCV approximation
# Purpose: after a signal becomes VALIDATED, estimate remaining runway to the
# nearest significant opposing High-Volume Node (HVN) and display it as %.
VP_ENABLE_RUNWAY = True
VP_CONTEXT_BARS = 180          # context window used to build the volume-at-price profile
VP_MIN_CONTEXT_BARS = 80       # minimum bars required to compute a stable VP runway
VP_BINS_MIN = 32
VP_BINS_MAX = 96
VP_BIN_ATR_FRACTION = 0.25     # target price-bin size ~= 0.25 * median ATR (context)
VP_BIN_PCT_FLOOR = 0.0025      # but never smaller than 0.25% of price
VP_SMOOTH_KERNEL = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
VP_PEAK_REL_MAX_MIN = 0.18     # peak must be >= 18% of max smoothed profile
VP_CLUSTER_FLOOR_FRAC_PEAK = 0.35
VP_CLUSTER_FLOOR_REL_MAX = 0.08
VP_MIN_CLUSTER_MASS_FRAC = 0.05  # node must contain >= 5% of profile volume
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
def _json_default(o):
    """JSON serializer for objects not serializable by default json code."""
    try:
        import numpy as _np
        import pandas as _pd
        import datetime as _dt
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if isinstance(o, (_dt.datetime, _dt.date)):
            return o.isoformat()
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.bool_,)):
            return bool(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return str(o)

def save_state(state: Dict) -> None:
    write_text(STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False, default=_json_default))
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
def fetch_rss_headlines(limit_total: int = 14) -> List[Dict[str, str]]:
    """Fetch RSS headlines from a small set of popular sources.
    Note: Financial Times dropped (paywall/open issues). Yahoo Finance included via multiple feeds for robustness.
    """
    feeds = [
        ("Yahoo Finance Top Stories", "https://finance.yahoo.com/rss/topstories"),
        ("Yahoo Finance — S&P 500", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"),
        ("Yahoo Finance — Nasdaq", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US"),
        ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("CNBC Markets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters Top News", "https://feeds.reuters.com/reuters/topNews"),
        ("MarketWatch Top Stories", "https://feeds.marketwatch.com/marketwatch/topstories"),
        ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        ("The Guardian Business", "https://www.theguardian.com/uk/business/rss"),
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
    # Simple relevancy: keep items that look finance/markets related first.
    def score(it: Dict[str, str]) -> int:
        txt = ((it.get("title", "") or "") + " " + (it.get("summary", "") or "")).lower()
        hits = 0
        for k in ["earnings","guidance","fed","rates","inflation","jobs","cpi","pce","bond","yield","oil","opec",
                  "ai","chip","semiconductor","nvidia","tesla","apple","amazon","microsoft","google","meta",
                  "crypto","bitcoin","geopolit","sanction","tariff","china","europe","ukraine","gaza"]:
            if k in txt:
                hits += 1
        return hits
    uniq.sort(key=score, reverse=True)
    return uniq[:limit_total]
def _clean_ticker(t: str) -> str:
    t = str(t).strip()
    # Wikipedia uses BRK.B -> Yahoo uses BRK-B (share class dot → hyphen).
    # MUST NOT convert exchange suffixes like .L, .DE, .TO, .AX, .SW, etc.
    # Strategy: if the part after the dot is a known exchange suffix, leave it alone.
    _EXCHANGE_SUFFIXES = {
        # Developed markets
        "L", "DE", "TO", "AX", "SW", "MI", "HK", "MC", "CO", "SI", "AS",
        "OL", "TA", "VI", "NZ", "IR", "PA", "ST", "BR", "HE", "LS", "T",
        "V", "F",  # Toronto Venture (.V), Frankfurt (.F)
        # Korea
        "KS", "KQ",
        # EM — v92: full set so no EM suffix is ever mangled to a hyphen
        "SA", "JK", "BK", "IS", "NS", "BO",  # Brazil, Indonesia, Thailand, Turkey, India NSE/BSE
        "TW",        # Taiwan
        "SS", "SZ",  # China Shanghai / Shenzhen
        "JO",        # South Africa (Johannesburg)
        "MX",        # Mexico
        "KL",        # Malaysia
        "SR",        # Saudi Arabia
        "AD", "DU", "AE",  # UAE (Abu Dhabi / Dubai / unified .AE — remapped by universe.py)
        "WA",        # Poland (Warsaw)
        "AT",        # Greece (Athens)
        "CA",        # Egypt (Cairo)
        "PS",        # Philippines
        "TWO",       # Taiwan Gretai Securities Market
        "QA",        # Qatar
        "PR",        # Czech Republic (Prague)
        "BD",        # Hungary (Budapest)
        "SN",        # Chile (Santiago)
        "CL",        # Colombia
        "LM",        # Peru (Lima)
        "KA",        # Pakistan (Karachi)
    }
    if "." in t:
        parts = t.rsplit(".", 1)
        if len(parts) == 2 and parts[1].upper() in _EXCHANGE_SUFFIXES:
            return t  # exchange suffix — do NOT convert
    # Safe to convert: share class like BRK.B, BF.A, MOG.B
    if re.fullmatch(r"[A-Z]+[.][A-Z]+", t):
        return t.replace(".", "-")
    return t

# ── Ghost ticker guard ────────────────────────────────────────────────────────
# is_ghost_ticker() imported from universe.py — single definition, no duplication.

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
    # Always include the default  watchlist unless explicitly disabled.
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
# ----------------------------
# Watchlist sector overrides (S&P 500 11-sector taxonomy)
# ----------------------------
# Used when the local MSCI/Sector classification CSV does not contain a ticker.
# We still prefer the CSV when available.
WATCHLIST_SECTOR_OVERRIDES: Dict[str, str] = {
    # Information Technology (incl. semis, EDA, quantum, software/data)
    "NVDA": "Information Technology",
    "ARM": "Information Technology",
    "AVGO": "Information Technology",
    "TSM": "Information Technology",
    "000660.KS": "Information Technology",
    "ASML": "Information Technology",
    "AMAT": "Information Technology",
    "LRCX": "Information Technology",
    "SNPS": "Information Technology",
    "CDNS": "Information Technology",
    "IONQ": "Information Technology",
    "QBTS": "Information Technology",
    "PLTR": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "RRTL.DE": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "MELI": "Consumer Discretionary",
    "DASH": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "ANF": "Consumer Discretionary",
    "DECK": "Consumer Discretionary",
    "BYDDY": "Consumer Discretionary",
    "MC.PA": "Consumer Discretionary",
    "RMS.PA": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples",
    # Financials
    "HOOD": "Financials",
    "NU": "Financials",
    "PGR": "Financials",
    "MUV2.DE": "Financials",
    "UCG.MI": "Financials",
    # Real Estate
    "ARR": "Real Estate",
    # Health Care
    "ISRG": "Health Care",
    "LLY": "Health Care",
    "NVO": "Health Care",
    # Utilities (power & generators)
    "VST": "Utilities",
    "CEG": "Utilities",
    "OKLO": "Utilities",
    "SMR": "Utilities",
    # Materials / Industrials (uranium & fuel-cycle; best-effort mapping)
    "CCJ": "Materials",
    "LEU": "Industrials",
    # Energy (oil & refiners)
    "CVX": "Energy",
    "REP.MC": "Energy",
    "MAU.PA": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    # Industrials (shipping / transport)
    "NAT": "Industrials",
    "INSW": "Industrials",
    "TNK": "Industrials",
    "FRO": "Industrials",
}
WATCHLIST_SECTOR_BY_TICKER: Dict[str, str] = { _clean_ticker(k): v for k, v in WATCHLIST_SECTOR_OVERRIDES.items() }
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
    # Full ticker normalization via universe.py _normalize_ticker:
    # covers Turkey (-E.IS), Thailand (.R.BK), UAE (.AD/.DU→.AE),
    # London double-dot (BA..L), Canada/Chile class shares (RCI.B.TO→RCI-B.TO)
    from universe import _normalize_ticker as _uni_norm
    df["Ticker"] = df["Ticker"].map(_uni_norm)
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
def build_sector_resolver(msci_df: pd.DataFrame):
    """Resolve ticker -> S&P 11-sector label.
    Preference order:
      1) local MSCI/Sector classification CSV (more accurate when available)
      2) WATCHLIST_SECTOR_OVERRIDES fallback
      3) "Unclassified"
    """
    msci_sector: Dict[str, str] = {}
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
        base = _base_ticker(t)
        s = msci_sector.get(t) or msci_sector.get(base)
        if s:
            return s
        s2 = WATCHLIST_SECTOR_BY_TICKER.get(t) or WATCHLIST_SECTOR_BY_TICKER.get(base)
        if s2:
            return s2
        if t in COMMODITY_TICKERS or base in COMMODITY_TICKERS:
            return "Commodities"
        return "Unclassified"
    return _resolve
# Backward-compatible alias (older code paths)
build_sector_resolver = build_sector_resolver
def _infer_country_from_ticker(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    # Exchange suffix heuristics (best-effort for watchlist names not in MSCI map)
    suffix = ""
    if "." in t:
        suffix = t.rsplit(".", 1)[-1]
    suffix_map = {
        "KS":  "South Korea",
        "KQ":  "South Korea (KOSDAQ)",
        "T":   "Japan",
        "DE":  "Germany",
        "MI":  "Italy",
        "PA":  "France",
        "SW":  "Switzerland",
        "L":   "United Kingdom",
        "MC":  "Spain",
        "AS":  "Netherlands",
        "HK":  "Hong Kong",
        "TO":  "Canada",
        "AX":  "Australia",
        "NZ":  "New Zealand",
        "ST":  "Sweden",
        "CO":  "Denmark",
        "HE":  "Finland",
        "OL":  "Norway",
        "BR":  "Belgium",
        "IS":  "Turkey",
        "SA":  "Brazil",
        "NS":  "India (NSE)",
        "BO":  "India (BSE)",
        "SI":  "Singapore",
        "TW":  "Taiwan",
        "TWO": "Taiwan (Gretai)",
        "AE":  "UAE",
        "AD":  "UAE",
        "DU":  "UAE",
        "SR":  "Saudi Arabia",
        "QA":  "Qatar",
        "KW":  "Kuwait",
        "JO":  "South Africa",
        "MX":  "Mexico",
        "BK":  "Thailand",
        "JK":  "Indonesia",
        "SN":  "Chile",
        "WA":  "Poland",
        "AT":  "Greece",
        "VI":  "Austria",
        "IR":  "Ireland",
        "LS":  "Portugal",
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
        if t in COMMODITY_NAME_OVERRIDES or base in COMMODITY_NAME_OVERRIDES:
            return COMMODITY_NAME_OVERRIDES.get(t) or COMMODITY_NAME_OVERRIDES.get(base) or ""
        if t in NAME_OVERRIDES:
            return NAME_OVERRIDES[t]
        if base in NAME_OVERRIDES:
            return NAME_OVERRIDES[base]
        return msci_company.get(t) or msci_company.get(base) or _display_name(t)
    def _country(ticker: str) -> str:
        t = str(ticker or "").strip()
        if not t:
            return ""
        base = _base_ticker(t)
        if t in COMMODITY_TICKERS or base in COMMODITY_TICKERS:
            return ""
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
    """Deterministic OHLCV downloader — grouped by exchange.
    yfinance bulk downloads work best when all tickers share the same exchange.
    Mixed-exchange batches cause Yahoo's API to silently drop non-US tickers.
    Strategy:
      1) Group tickers by exchange suffix (.L, .TO, .DE, etc. — suffix-less = US)
      2) Download each exchange group in its own chunks
      3) Retry any still-missing tickers individually
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    tickers = sorted(dict.fromkeys(tickers))  # stable unique
    # Group by exchange suffix
    groups: Dict[str, List[str]] = {}
    for t in tickers:
        if "." in t:
            sfx = "." + t.rsplit(".", 1)[-1]
        else:
            sfx = "US"
        groups.setdefault(sfx, []).append(t)
    def _download_one(t: str) -> Optional[pd.DataFrame]:
        try:
            data = yf.download(
                tickers=[t],
                period=DOWNLOAD_PERIOD,
                interval=DOWNLOAD_INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            df = extract_ohlcv_from_download(data, t)
            return df if df is not None and not df.empty else None
        except Exception:
            return None
    # Download each exchange group in chunks
    n_groups = len(groups)
    for gi, (sfx, group_tickers) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1]))):
        for i in range(0, len(group_tickers), CHUNK_SIZE):
            chunk = group_tickers[i:i + CHUNK_SIZE]
            if not chunk:
                continue
            data = None
            try:
                data = yf.download(
                    tickers=chunk,
                    period=DOWNLOAD_PERIOD,
                    interval=DOWNLOAD_INTERVAL,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,
                    progress=False,
                )
            except Exception:
                data = None
            if data is not None and not getattr(data, "empty", True):
                for t in chunk:
                    df = extract_ohlcv_from_download(data, t)
                    if df is not None and not df.empty:
                        out[t] = df
            # Retry missing from this chunk individually
            missing = [t for t in chunk if t not in out]
            for t in missing:
                df = _download_one(t)
                if df is not None and not df.empty:
                    out[t] = df
        got = sum(1 for t in group_tickers if t in out)
        print(f"[download] {sfx}: {got}/{len(group_tickers)} downloaded")
    # Final retry pass for all still-missing
    still_missing = [t for t in tickers if t not in out]
    if still_missing:
        print(f"[download] final retry for {len(still_missing)} still-missing tickers")
        for t in still_missing:
            df = _download_one(t)
            if df is not None and not df.empty:
                out[t] = df
        recovered = len(still_missing) - len([t for t in still_missing if t not in out])
        if recovered:
            print(f"[download] final retry recovered {recovered}/{len(still_missing)}")
    print(f"[download] total: {len(out)}/{len(tickers)} tickers downloaded")
    return out
def pct_change_last(df: pd.DataFrame) -> Optional[float]:
    c = df["Close"].dropna()
    if len(c) < 2:
        return None
    return float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
def _clv_at_bar(d: pd.DataFrame, i: int) -> float:
    """Close Location Value in [-1,+1] for bar i."""
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
    if sig.startswith("VALIDATED_"):
        return 4
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
    df_conf_new: pd.DataFrame,
    df_conf_old: pd.DataFrame,
    df_val_new: pd.DataFrame,
    df_val_old: pd.DataFrame,
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
    rows += _iter_df(df_conf_new, True)
    rows += _iter_df(df_conf_old, False)
    rows += _iter_df(df_val_new, True)
    rows += _iter_df(df_val_old, False)
    sigs_by_t: Dict[str, List[Tuple[str, float, bool]]] = {}
    for t, s, dist, is_new in rows:
        if not t or not s:
            continue
        sigs_by_t.setdefault(t, []).append((s, 0.0 if math.isnan(dist) else dist, is_new))
    cat_stats = {}
    for cat, tickers in watchlist_groups.items():
        counts = {"VALID_UP": 0, "VALID_DN": 0, "CONF_UP": 0, "CONF_DN": 0, "EARLY_UP": 0, "EARLY_DN": 0}
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
            stage = "EARLY"
            if sig.startswith("VALIDATED_"):
                stage = "VALID"
            elif sig.startswith("CONFIRMED_"):
                stage = "CONF"
            key = stage + ("_UP" if d > 0 else "_DN")
            counts[key] += 1
            score += w * d
            label = ticker_labels.get(t, t)
            leaders.append((w, 1 if is_new else 0, abs(dist), label, sig))
        leaders.sort(reverse=True)
        top = [{"ticker": x[3], "signal": x[4]} for x in leaders[:3]]
        cat_stats[cat] = {"score": score, "counts": counts, "top": top}
    md = []
    md.append("### 4A) Watchlist emerging chart trends")
    md.append("")
    md.append("_Logic: score each ticker by stage (EARLY=1, CONFIRMED=3, VALIDATED=4) × direction (BREAKOUT=+1, BREAKDOWN=-1), then aggregate by sector._")
    md.append("")
    # Order: EARLY -> CONFIRMED -> VALIDATED
    md.append("| Sector | Bias | EARLY↑ | EARLY↓ | CONF↑ | CONF↓ | VALID↑ | VALID↓ |")
    md.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cat, s in cat_stats.items():
        sc = s["score"]
        bias = "Bullish" if sc >= 3 else "Bearish" if sc <= -3 else "Mixed"
        c = s["counts"]
        md.append(
            f"| {cat} | {bias} | {c.get('EARLY_UP',0)} | {c.get('EARLY_DN',0)} | {c.get('CONF_UP',0)} | {c.get('CONF_DN',0)} | {c.get('VALID_UP',0)} | {c.get('VALID_DN',0)} |"
        )
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
    lines.append(f"Daily Ticker Report — {header_time} ({SCAN_VERSION})")
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
def yahoo_quote(symbols: List[str]) -> List[Dict]:
    """Fetch Yahoo Finance quote data (regular + extended hours) via the public quote endpoint.
    Robustness:
      - URL-encodes symbols safely
      - Retries on transient HTTP errors (429/5xx)
      - Falls back to per-symbol requests if a chunk fails
    """
    if not symbols:
        return []
    from urllib.parse import quote
    import time
    def _fetch(sym_list: List[str]) -> List[Dict]:
        if not sym_list:
            return []
        sym_str = ",".join([str(s).strip() for s in sym_list if str(s).strip()])
        if not sym_str:
            return []
        # Keep commas unescaped; escape everything else safely.
        url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + quote(sym_str, safe=",")
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))
        res = (((data or {}).get("quoteResponse") or {}).get("result")) or []
        if isinstance(res, list):
            return [x for x in res if isinstance(x, dict)]
        return []
    out: List[Dict] = []
    CH = 50  # smaller chunks are less likely to be throttled
    for i in range(0, len(symbols), CH):
        chunk = symbols[i:i + CH]
        ok = False
        for attempt in range(3):
            try:
                out.extend(_fetch(chunk))
                ok = True
                break
            except HTTPError as e:
                # transient throttling / gateway errors
                code = getattr(e, "code", None)
                if code in (429, 502, 503, 504):
                    time.sleep(0.6 + 0.7 * attempt)
                    continue
                print(f"[yahoo_quote] HTTP {code} for chunk starting {chunk[0]} (attempt {attempt+1})")
                break
            except Exception as exc:
                print(f"[yahoo_quote] error for chunk starting {chunk[0]}: {type(exc).__name__}: {exc}")
                time.sleep(0.3 + 0.4 * attempt)
                continue
        if not ok and len(chunk) > 1:
            # Fallback: per-symbol
            for s in chunk:
                try:
                    out.extend(_fetch([s]))
                except Exception:
                    continue
    print(f"[yahoo_quote] requested {len(symbols)} symbols, got {len(out)} quote records")
    return out
def fetch_watchlist_afterhours_movers_yahoo(symbols: List[str]) -> pd.DataFrame:
    """Compute AFTER-HOURS % moves using yfinance Ticker.info (authenticated).
    The old v7/finance/quote endpoint is deprecated and returns 403 in many environments.
    Primary: yfinance Ticker.info → postMarketPrice / regularMarketPrice
    Fallback: raw v7 endpoint (may still work in some environments)
    Output schema: ['symbol','pct'] where pct is in percent points (e.g., +10.2 for +10.2%).
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "pct"])
    rows = []
    # Primary: yfinance Ticker.info (authenticated, reliable)
    for sym in symbols:
        try:
            tk = yf.Ticker(str(sym).strip())
            info = tk.info if hasattr(tk, "info") else {}
            if not isinstance(info, dict) or not info:
                continue
            reg_price = info.get("regularMarketPrice") or info.get("currentPrice")
            post_price = info.get("postMarketPrice")
            pre_price = info.get("preMarketPrice")
            pct = info.get("postMarketChangePercent")
            if pct is not None:
                pct_f = float(pct)
                # Yahoo sometimes returns fractional (0.04) instead of percent (4.0)
                if abs(pct_f) < 1.0 and post_price and reg_price and float(reg_price) > 0:
                    alt = (float(post_price) / float(reg_price) - 1.0) * 100.0
                    if abs(alt) > abs(pct_f) * 5:
                        pct_f = alt
                rows.append({"symbol": sym, "pct": pct_f})
            elif post_price is not None and reg_price not in (None, 0, 0.0):
                pct_f = (float(post_price) / float(reg_price) - 1.0) * 100.0
                rows.append({"symbol": sym, "pct": pct_f})
            elif pre_price is not None and reg_price not in (None, 0, 0.0):
                pct_f = (float(pre_price) / float(reg_price) - 1.0) * 100.0
                rows.append({"symbol": sym, "pct": pct_f})
        except Exception:
            continue
    if rows:
        print(f"[after-hours] yfinance Ticker.info: got data for {len(rows)}/{len(symbols)} symbols")
    # Fallback: raw v7 endpoint (may still work in some environments)
    if len(rows) < len(symbols) * 0.5:
        try:
            v7_rows = _yahoo_quote_v7_afterhours(symbols)
            if v7_rows:
                have = {r["symbol"] for r in rows}
                for r in v7_rows:
                    if r["symbol"] not in have:
                        rows.append(r)
                print(f"[after-hours] v7 fallback added {len(v7_rows)} records")
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=["symbol", "pct"])
    return pd.DataFrame(rows)[["symbol", "pct"]]

def _yahoo_quote_v7_afterhours(symbols: List[str]) -> List[Dict]:
    """Legacy v7/finance/quote endpoint (may return 403 — kept as fallback)."""
    q = yahoo_quote(symbols or [])
    rows = []
    for it in q:
        try:
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue
            reg_price = it.get("regularMarketPrice")
            post_price = it.get("postMarketPrice")
            pct = it.get("postMarketChangePercent")
            if pct is not None:
                rows.append({"symbol": sym, "pct": float(pct)})
            elif post_price is not None and reg_price not in (None, 0, 0.0):
                pct = (float(post_price) / float(reg_price) - 1.0) * 100.0
                rows.append({"symbol": sym, "pct": pct})
        except Exception:
            continue
    return rows
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
    Always returns schema ['symbol','pct'] and preserves an existing numeric 'pct' column
    (e.g., Yahoo quote-based after-hours movers).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "pct"])
    out = df.copy()
    # Preserve existing pct when present
    if "pct" in out.columns and "_pct" not in out.columns:
        out["pct"] = pd.to_numeric(out["pct"], errors="coerce")
    elif "_pct" in out.columns:
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
    # Sort by absolute move (biggest first)
    out = out.sort_values("pct", ascending=False, key=lambda s: s.abs())
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
            return f"## 3) Earnings next {days} days (your watchlist)\n\n_None from watchlist in the next {days} days._\n"
        # Render as markdown table (right-align numeric)
        md = []
        md.append(f"## 3) Earnings next {days} days (your watchlist)\n")
        md.append("_Upcoming earnings dates for your  watchlist._\n")
        md.append(md_table_from_df(df, cols=["Ticker", "Earnings Date", "Days"]))
        return "\n".join(md) + "\n"
    except Exception:
        return f"## 3) Earnings next {days} days (your watchlist)\n\n_(Failed to fetch earnings calendar.)_\n"
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
    stage_status: Optional[str] = None
    stage_age_bars: Optional[int] = None
    breakout_start: Optional[str] = None
    pct_today: Optional[float] = None
    clv: Optional[float] = None
    vol_ratio: Optional[float] = None
    chart_path: Optional[str] = None
    vp_hvn_runway_pct: Optional[float] = None
    vp_hvn_zone_low: Optional[float] = None
    vp_hvn_zone_high: Optional[float] = None
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
def _vp_context_slice(d: pd.DataFrame, end_idx: Optional[int] = None, lookback: int = VP_CONTEXT_BARS) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame()
    n = len(d)
    if end_idx is None:
        end_idx = n - 1
    end_idx = int(max(0, min(end_idx, n - 1)))
    start_idx = max(0, end_idx - int(lookback) + 1)
    out = d.iloc[start_idx : end_idx + 1].copy()
    return out
def _vp_build_histogram_daily(context: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Build a deterministic daily-OHLCV volume-at-price approximation.
    Implementation choice (speed + consistency): weight each bar's typical price (HLC3)
    by traded volume. This is an approximation (not tick-level volume profile), but it is
    stable enough for cross-sectional screening and backtests.
    """
    if context is None or context.empty:
        return None
    req = [c for c in ("High", "Low", "Close", "Volume") if c in context.columns]
    if len(req) < 4:
        return None
    c = context.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    if len(c) < VP_MIN_CONTEXT_BARS:
        return None
    hi = pd.to_numeric(c["High"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(c["Low"], errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(c["Close"], errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(c["Volume"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl) & np.isfinite(vol) & (vol > 0)
    if mask.sum() < VP_MIN_CONTEXT_BARS:
        return None
    hi = hi[mask]
    lo = lo[mask]
    cl = cl[mask]
    vol = vol[mask]
    tp = (hi + lo + cl) / 3.0
    pmin = float(np.nanmin(lo))
    pmax = float(np.nanmax(hi))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return None
    try:
        a_ctx = atr(c, ATR_N)
        med_atr = float(pd.to_numeric(a_ctx, errors="coerce").dropna().median()) if a_ctx is not None else float("nan")
    except Exception:
        med_atr = float("nan")
    last_close = float(cl[-1]) if len(cl) else float("nan")
    if not np.isfinite(last_close) or last_close <= 0:
        last_close = max((pmin + pmax) / 2.0, 1e-6)
    bin_size = float("nan")
    if np.isfinite(med_atr) and med_atr > 0:
        bin_size = max(med_atr * VP_BIN_ATR_FRACTION, last_close * VP_BIN_PCT_FLOOR)
    else:
        bin_size = max(last_close * VP_BIN_PCT_FLOOR, (pmax - pmin) / 50.0)
    price_range = max(pmax - pmin, 1e-9)
    bins_n = int(math.ceil(price_range / max(bin_size, 1e-9)))
    bins_n = int(max(VP_BINS_MIN, min(VP_BINS_MAX, bins_n)))
    edges = np.linspace(pmin, pmax, bins_n + 1)
    hist_raw, _ = np.histogram(tp, bins=edges, weights=vol)
    if hist_raw.size == 0 or not np.isfinite(hist_raw).any() or np.nansum(hist_raw) <= 0:
        return None
    k = VP_SMOOTH_KERNEL.astype(float)
    if np.nansum(k) <= 0:
        k = np.array([1.0], dtype=float)
    k = k / np.nansum(k)
    hist_smooth = np.convolve(hist_raw.astype(float), k, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "edges": edges,
        "centers": centers,
        "hist_raw": hist_raw.astype(float),
        "hist_smooth": hist_smooth.astype(float),
        "total_vol": float(np.nansum(hist_raw)),
        "bin_size": float((edges[1] - edges[0]) if len(edges) >= 2 else np.nan),
    }
def _vp_detect_hvn_zones(profile: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not profile:
        return []
    raw = np.asarray(profile.get("hist_raw", []), dtype=float)
    sm = np.asarray(profile.get("hist_smooth", []), dtype=float)
    edges = np.asarray(profile.get("edges", []), dtype=float)
    centers = np.asarray(profile.get("centers", []), dtype=float)
    total_vol = float(profile.get("total_vol", 0.0) or 0.0)
    if raw.size < 3 or sm.size != raw.size or centers.size != raw.size or edges.size != raw.size + 1 or total_vol <= 0:
        return []
    sm = np.nan_to_num(sm, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    sm_max = float(np.max(sm)) if sm.size else 0.0
    if sm_max <= 0:
        return []
    # Candidate peaks: local maxima in smoothed profile above relative threshold
    peaks: List[int] = []
    peak_floor = VP_PEAK_REL_MAX_MIN * sm_max
    for i in range(1, len(sm) - 1):
        if sm[i] < peak_floor:
            continue
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            if sm[i] > sm[i - 1] or sm[i] > sm[i + 1]:
                peaks.append(i)
    if not peaks:
        i = int(np.argmax(sm))
        peaks = [i] if sm[i] > 0 else []
    if not peaks:
        return []
    zones: List[Dict[str, float]] = []
    for p in peaks:
        peak_val = float(sm[p])
        floor_val = max(VP_CLUSTER_FLOOR_FRAC_PEAK * peak_val, VP_CLUSTER_FLOOR_REL_MAX * sm_max)
        l = p
        while l - 1 >= 0 and sm[l - 1] >= floor_val:
            l -= 1
        r = p
        while r + 1 < len(sm) and sm[r + 1] >= floor_val:
            r += 1
        mass = float(np.sum(raw[l:r + 1]))
        mass_frac = mass / total_vol if total_vol > 0 else 0.0
        if mass_frac < VP_MIN_CLUSTER_MASS_FRAC:
            continue
        zones.append({
            "peak": float(centers[p]),
            "peak_val": float(raw[p]),
            "smooth_peak": peak_val,
            "low": float(edges[l]),
            "high": float(edges[r + 1]),
            "mass": mass,
            "mass_frac": float(mass_frac),
            "i_l": float(l),
            "i_r": float(r),
            "i_p": float(p),
        })
    if not zones:
        return []
    # Merge overlapping zones (keep stronger peak and combine mass/range)
    zones = sorted(zones, key=lambda z: (z["low"], z["high"]))
    merged: List[Dict[str, float]] = []
    for z in zones:
        if not merged or z["low"] > merged[-1]["high"]:
            merged.append(dict(z))
            continue
        m = merged[-1]
        # overlap -> merge ranges and mass; keep stronger peak label
        m["low"] = min(m["low"], z["low"])
        m["high"] = max(m["high"], z["high"])
        m["mass"] = float(m.get("mass", 0.0) + z.get("mass", 0.0))
        m["mass_frac"] = float(m.get("mass_frac", 0.0) + z.get("mass_frac", 0.0))
        if z.get("smooth_peak", 0.0) > m.get("smooth_peak", 0.0):
            for k in ("peak", "peak_val", "smooth_peak", "i_p"):
                m[k] = z.get(k, m.get(k))
    return merged
def _vp_nearest_opposing_hvn_zone(d: pd.DataFrame, close: float, direction: str, end_idx: Optional[int] = None) -> Optional[Dict[str, float]]:
    if not VP_ENABLE_RUNWAY:
        return None
    context = _vp_context_slice(d, end_idx=end_idx, lookback=VP_CONTEXT_BARS)
    profile = _vp_build_histogram_daily(context)
    zones = _vp_detect_hvn_zones(profile)
    if not zones:
        return None
    direction = str(direction or "").upper()
    if direction == "BREAKOUT":
        # opposing node is the first significant overhead HVN zone. Use zone lower bound as the wall start.
        overhead = [z for z in zones if float(z.get("high", np.nan)) > close]
        if not overhead:
            return None
        overhead.sort(key=lambda z: (max(float(z.get("low", np.inf)), close) - close, float(z.get("low", np.inf))))
        return overhead[0]
    elif direction == "BREAKDOWN":
        below = [z for z in zones if float(z.get("low", np.nan)) < close]
        if not below:
            return None
        # nearest opposing support below: zone upper bound closest below current price
        below.sort(key=lambda z: (close - min(float(z.get("high", -np.inf)), close), -float(z.get("high", -np.inf))))
        return below[0]
    return None
def _vp_runway_to_hvn_pct(d: pd.DataFrame, close: float, direction: str, end_idx: Optional[int] = None) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """Return signed runway % in the signal direction to nearest opposing HVN zone.
    Longs (BREAKOUT):  ((zone_low  - close) / close) * 100
    Shorts (BREAKDOWN):((close - zone_high) / close) * 100
    Positive => runway remains. Negative => price is already inside/past the HVN wall.
    """
    try:
        close = float(close)
    except Exception:
        return None, None
    if not np.isfinite(close) or close <= 0:
        return None, None
    z = _vp_nearest_opposing_hvn_zone(d, close=close, direction=direction, end_idx=end_idx)
    if not z:
        return None, None
    direction = str(direction or "").upper()
    try:
        if direction == "BREAKOUT":
            wall = float(z.get("low"))
            pct = ((wall - close) / close) * 100.0
        elif direction == "BREAKDOWN":
            wall = float(z.get("high"))
            pct = ((close - wall) / close) * 100.0
        else:
            return None, z
        if not np.isfinite(pct):
            return None, z
        return float(pct), z
    except Exception:
        return None, z
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
def _diagnose_swing_high(
    df: pd.DataFrame,
    ts: str,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
) -> Dict[str, Any]:
    """Explain deterministically why a given bar *is / is not* a swing-high pivot under _swing_points_ohlc()."""
    out: Dict[str, Any] = {"ts_req": ts, "ok": False}
    try:
        dd0 = df.dropna(subset=["High", "Low", "Close"]).copy()
        if dd0.empty:
            out["reason"] = "empty_df_after_dropna"
            return out
        # Use the SAME calendar-day window as detection (CHART_WINDOW_DAYS), so indices match detector logic.
        if isinstance(dd0.index, pd.DatetimeIndex):
            cutoff = dd0.index[-1] - pd.Timedelta(days=CHART_WINDOW_DAYS)
            dd = dd0.loc[dd0.index >= cutoff].copy()
        else:
            dd = dd0.tail(LOOKBACK_DAYS).copy()
        dd = _latest_completed_close_df(dd)
        if dd.empty:
            out["reason"] = "empty_df_after_dropna"
            return out
        t_req = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t_req):
            out["reason"] = "bad_ts"
            return out
        # Find exact match; else choose nearest index (deterministic).
        if isinstance(dd.index, pd.DatetimeIndex):
            try:
                i = int(dd.index.get_loc(t_req))
                t_hit = dd.index[i]
            except Exception:
                # nearest
                diffs = np.abs((dd.index - t_req).astype("timedelta64[s]").astype(np.int64))
                i = int(np.nanargmin(diffs))
                t_hit = dd.index[i]
                out["nearest"] = True
        else:
            out["reason"] = "non_datetime_index"
            return out
        out["ts_hit"] = str(t_hit)
        out["i"] = int(i)
        if i < window or i >= len(dd) - window:
            out["reason"] = "too_close_to_edges_for_window"
            return out
        hi = dd["High"].astype(float).values
        lo = dd["Low"].astype(float).values
        hwin = hi[i - window : i + window + 1]
        lwin = lo[i - window : i + window + 1]
        out["hwin_max"] = float(np.max(hwin))
        out["hwin_max_count"] = int(np.sum(hwin == hi[i]))
        out["hi_i"] = float(hi[i])
        out["lo_i"] = float(lo[i])
        out["close_i"] = float(dd["Close"].iloc[i])
        # ATR at i (mirror logic in _swing_points_ohlc)
        atr_s = atr(dd, ATR_N)
        atr_v = pd.to_numeric(atr_s, errors="coerce").values if atr_s is not None else np.full(len(dd), np.nan)
        atr_i = atr_v[i] if i < len(atr_v) and np.isfinite(atr_v[i]) else np.nan
        if not np.isfinite(atr_i):
            lo_i = max(0, i - 20)
            med = np.nanmedian(atr_v[lo_i : i + 1])
            atr_i = med if np.isfinite(med) else np.nan
        if not np.isfinite(atr_i):
            atr_i = max(float(np.nanmedian((hwin - lwin))), 1e-6)
        out["atr_i"] = float(atr_i)
        is_local_max = bool(hi[i] == np.max(hwin))
        is_unique = bool(np.sum(hwin == hi[i]) == 1)
        prominence = float(hi[i] - np.min(lwin))
        thresh = float(prominence_atr_mult * atr_i)
        out["is_local_max"] = is_local_max
        out["is_unique_max"] = is_unique
        out["prominence"] = float(prominence)
        out["prom_thresh"] = float(thresh)
        out["prom_ok"] = bool(prominence >= thresh)
        out["ok"] = bool(is_local_max and is_unique and prominence >= thresh)
        if not out["ok"]:
            reasons = []
            if not is_local_max:
                reasons.append("not_local_max_in_window")
            if is_local_max and (not is_unique):
                reasons.append("ties_for_local_max")
            if prominence < thresh:
                reasons.append("prominence_below_threshold")
            out["reason"] = ",".join(reasons) if reasons else "unknown"
        return out
    except Exception as e:
        out["reason"] = f"exception:{type(e).__name__}"
        return out
def _swing_highs_on_close(
    df: pd.DataFrame,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
    allow_tie_high_2dp: bool = True,
) -> List[int]:
    """
    Close-based swing highs (for HS shoulders/head in a Close-driven deterministic model).
    - local max on Close in [i-window, i+window]
    - unique max (or optional 2dp tie handling)
    - prominence measured vs window Low using ATR filter
    """
    dd = df.dropna(subset=["High", "Low", "Close"]).copy()
    if dd.empty or len(dd) < (2 * window + 5):
        return []
    cl = dd["Close"].astype(float).values
    lo = dd["Low"].astype(float).values
    atr_s = atr(dd, ATR_N)
    atr_v = pd.to_numeric(atr_s, errors="coerce").values if atr_s is not None else np.full(len(dd), np.nan)
    highs: List[int] = []
    for i in range(window, len(dd) - window):
        cwin = cl[i - window : i + window + 1]
        lwin = lo[i - window : i + window + 1]
        if np.isnan(cwin).any() or np.isnan(lwin).any():
            continue
        atr_i = atr_v[i] if i < len(atr_v) and np.isfinite(atr_v[i]) else np.nan
        if not np.isfinite(atr_i):
            lo_i = max(0, i - 20)
            med = np.nanmedian(atr_v[lo_i : i + 1])
            atr_i = med if np.isfinite(med) else np.nan
        if not np.isfinite(atr_i):
            atr_i = max(float(np.nanmedian((np.nanmax(dd["High"].values[i-window:i+window+1]) - lwin))), 1e-6)
        is_local_max = bool(cl[i] == np.max(cwin))
        if not is_local_max:
            continue
        is_unique = bool(np.sum(cwin == cl[i]) == 1)
        ok_tie = False
        if (not is_unique) and allow_tie_high_2dp:
            try:
                mx2 = round(float(np.max(cwin)), 2)
                ties = [j for j, v in enumerate(cwin) if round(float(v), 2) == mx2]
                if ties:
                    chosen_global = (i - window) + int(ties[0])  # earliest tie rep
                    ok_tie = (i == chosen_global) and (round(float(cl[i]), 2) == mx2)
            except Exception:
                ok_tie = False
        if is_unique or ok_tie:
            prominence = float(cl[i] - np.min(lwin))
            if prominence >= float(prominence_atr_mult * atr_i):
                highs.append(i)
    return highs
def _swing_lows_on_close(
    df: pd.DataFrame,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
    allow_tie_low_2dp: bool = True,
) -> List[int]:
    """
    Close-based swing lows (for IHS shoulders/head in a Close-driven deterministic model).
    - local min on Close in [i-window, i+window]
    - unique min (or optional 2dp tie handling)
    - prominence measured vs window High using ATR filter
    """
    dd = df.dropna(subset=["High", "Low", "Close"]).copy()
    if dd.empty or len(dd) < (2 * window + 5):
        return []
    cl = dd["Close"].astype(float).values
    hi = dd["High"].astype(float).values
    atr_s = atr(dd, ATR_N)
    atr_v = pd.to_numeric(atr_s, errors="coerce").values if atr_s is not None else np.full(len(dd), np.nan)
    lows: List[int] = []
    for i in range(window, len(dd) - window):
        cwin = cl[i - window : i + window + 1]
        hwin = hi[i - window : i + window + 1]
        if np.isnan(cwin).any() or np.isnan(hwin).any():
            continue
        atr_i = atr_v[i] if i < len(atr_v) and np.isfinite(atr_v[i]) else np.nan
        if not np.isfinite(atr_i):
            lo_i = max(0, i - 20)
            med = np.nanmedian(atr_v[lo_i : i + 1])
            atr_i = med if np.isfinite(med) else np.nan
        if not np.isfinite(atr_i):
            atr_i = max(float(np.nanmedian((hwin - np.nanmin(dd["Low"].values[i-window:i+window+1])))), 1e-6)
        is_local_min = bool(cl[i] == np.min(cwin))
        if not is_local_min:
            continue
        is_unique = bool(np.sum(cwin == cl[i]) == 1)
        ok_tie = False
        if (not is_unique) and allow_tie_low_2dp:
            try:
                mn2 = round(float(np.min(cwin)), 2)
                ties = [j for j, v in enumerate(cwin) if round(float(v), 2) == mn2]
                if ties:
                    chosen_global = (i - window) + int(ties[0])  # earliest tie rep
                    ok_tie = (i == chosen_global) and (round(float(cl[i]), 2) == mn2)
            except Exception:
                ok_tie = False
        if is_unique or ok_tie:
            prominence = float(np.max(hwin) - cl[i])
            if prominence >= float(prominence_atr_mult * atr_i):
                lows.append(i)
    return lows
def _diagnose_swing_high_close(
    df: pd.DataFrame,
    ts: str,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
) -> Dict[str, Any]:
    """Explain deterministically why a given bar is / is not a Close-based swing high under _swing_highs_on_close()."""
    out: Dict[str, Any] = {"ts_req": ts, "ok": False}
    try:
        dd0 = df.dropna(subset=["High", "Low", "Close"]).copy()
        if dd0.empty:
            out["reason"] = "empty_df_after_dropna"
            return out
        if isinstance(dd0.index, pd.DatetimeIndex):
            cutoff = dd0.index[-1] - pd.Timedelta(days=CHART_WINDOW_DAYS)
            dd = dd0.loc[dd0.index >= cutoff].copy()
        else:
            dd = dd0.tail(LOOKBACK_DAYS).copy()
        dd = _latest_completed_close_df(dd)
        if dd.empty:
            out["reason"] = "empty_df_after_dropna"
            return out
        t_req = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t_req):
            out["reason"] = "bad_ts"
            return out
        if not isinstance(dd.index, pd.DatetimeIndex):
            out["reason"] = "non_datetime_index"
            return out
        try:
            i = int(dd.index.get_loc(t_req))
            t_hit = dd.index[i]
        except Exception:
            diffs = np.abs((dd.index - t_req).astype("timedelta64[s]").astype(np.int64))
            i = int(np.nanargmin(diffs))
            t_hit = dd.index[i]
            out["nearest"] = True
        out["ts_hit"] = str(t_hit)
        out["i"] = int(i)
        if i < window or i >= len(dd) - window:
            out["reason"] = "too_close_to_edges_for_window"
            return out
        cl = dd["Close"].astype(float).values
        lo = dd["Low"].astype(float).values
        cwin = cl[i - window : i + window + 1]
        lwin = lo[i - window : i + window + 1]
        out["cwin_max"] = float(np.max(cwin))
        out["cwin_max_count"] = int(np.sum(cwin == cl[i]))
        out["close_i"] = float(cl[i])
        out["low_i"] = float(lo[i])
        atr_s = atr(dd, ATR_N)
        atr_v = pd.to_numeric(atr_s, errors="coerce").values if atr_s is not None else np.full(len(dd), np.nan)
        atr_i = atr_v[i] if i < len(atr_v) and np.isfinite(atr_v[i]) else np.nan
        if not np.isfinite(atr_i):
            lo_i = max(0, i - 20)
            med = np.nanmedian(atr_v[lo_i : i + 1])
            atr_i = med if np.isfinite(med) else np.nan
        if not np.isfinite(atr_i):
            atr_i = max(float(np.nanmedian((dd["High"].astype(float).values[i-window:i+window+1] - lwin))), 1e-6)
        out["atr_i"] = float(atr_i)
        is_local_max = bool(cl[i] == np.max(cwin))
        is_unique = bool(np.sum(cwin == cl[i]) == 1)
        prominence = float(cl[i] - np.min(lwin))
        thresh = float(prominence_atr_mult * atr_i)
        out["is_local_max"] = is_local_max
        out["is_unique_max"] = is_unique
        out["prominence"] = float(prominence)
        out["prom_thresh"] = float(thresh)
        out["prom_ok"] = bool(prominence >= thresh)
        out["ok"] = bool(is_local_max and is_unique and prominence >= thresh)
        if not out["ok"]:
            reasons = []
            if not is_local_max:
                reasons.append("not_local_max_in_window")
            if is_local_max and (not is_unique):
                reasons.append("ties_for_local_max")
            if prominence < thresh:
                reasons.append("prominence_below_threshold")
            out["reason"] = ",".join(reasons) if reasons else "unknown"
        return out
    except Exception as e:
        out["reason"] = f"exception:{type(e).__name__}"
        return out
def _swing_points_ohlc(
    df: pd.DataFrame,
    window: int = 3,
    prominence_atr_mult: float = 0.5,
    allow_tie_low_2dp: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Deterministic pivots on High/Low with prominence filter:
    - swing high: local max in [i-window, i+window], unique, prominence >= 0.5 ATR
    - swing low: local min analogously (optionally allow ties at 2dp for HS neckline shelves)
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
        if lo[i] == np.min(lwin):
            ok_unique = (np.sum(lwin == lo[i]) == 1)
            ok_tie = False
            if (not ok_unique) and allow_tie_low_2dp:
                # HS exception: allow shelf/plateau lows (ties) if they match at 2 decimals.
                # Keep exactly ONE deterministic representative: the earliest bar in the tied-min set.
                try:
                    min2 = round(float(np.min(lwin)), 2)
                    ties = [j for j, v in enumerate(lwin) if round(float(v), 2) == min2]
                    if ties:
                        chosen_global = (i - window) + int(ties[0])
                        ok_tie = (i == chosen_global) and (round(float(lo[i]), 2) == min2)
                except Exception:
                    ok_tie = False
            if ok_unique or ok_tie:
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
def _after_close_cutoff_berlin(now: Optional[dt.datetime] = None) -> bool:
    """Simple rule: if local Berlin time >= 22:10, assume the latest daily candle is closed."""
    try:
        tz = ZoneInfo("Europe/Berlin")
        now2 = now or dt.datetime.now(tz)
    except Exception:
        now2 = now or dt.datetime.now()
    return (now2.hour, now2.minute) >= (22, 10)
def _latest_completed_close_df(d: pd.DataFrame) -> pd.DataFrame:
    """Return df sliced to the latest completed daily close.
    Rule:
      - If the last bar date is BEFORE today (Berlin), it is already a completed close -> keep it (weekends/holidays stable).
      - If the last bar date is today and Berlin time < 22:10 -> drop the last bar (intraday partial daily bar).
      - Otherwise keep.
    """
    if d is None or d.empty:
        return d
    if not isinstance(d.index, pd.DatetimeIndex):
        return d
    try:
        tz = ZoneInfo("Europe/Berlin")
        now = dt.datetime.now(tz)
        today = now.date()
        last_date = pd.Timestamp(d.index[-1]).date()
        if last_date < today:
            return d
        # last_date == today (or future)
        if last_date == today and (now.hour, now.minute) < (22, 10):
            return d.iloc[:-1].copy() if len(d) > 1 else d
        return d
    except Exception:
        return d
def _point_meta(df: pd.DataFrame, i: int, price: float, label: str, kind: str = "point") -> Dict[str, Any]:
    return {"t": _iso_ts(df.index[i]), "p": float(price), "label": str(label), "kind": kind, "i": int(i)}
def _line_meta(df: pd.DataFrame, i1: int, y1: float, i2: int, y2: float, label: str) -> Dict[str, Any]:
    return {
        "t1": _iso_ts(df.index[i1]), "y1": float(y1),
        "t2": _iso_ts(df.index[i2]), "y2": float(y2),
        "label": str(label), "i1": int(i1), "i2": int(i2)
    }
def _reindex_meta_to_df(meta: Dict[str, Any], d: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Re-map meta indices onto the current df slice using timestamps in meta ("t","t1","t2")."""
    if meta is None or not isinstance(meta, dict) or d is None or d.empty:
        return None
    if not isinstance(d.index, pd.DatetimeIndex):
        return None
    date_to_pos: Dict[str, int] = {}
    for pos, ts in enumerate(d.index):
        try:
            date_to_pos[pd.Timestamp(ts).date().isoformat()] = int(pos)
        except Exception:
            pass
    def _pos_from_iso(iso: Any) -> Optional[int]:
        try:
            t = pd.to_datetime(str(iso), utc=True, errors="coerce")
            if pd.isna(t):
                t = pd.to_datetime(str(iso), errors="coerce")
            if pd.isna(t):
                return None
            k = t.date().isoformat()
            return int(date_to_pos[k]) if k in date_to_pos else None
        except Exception:
            return None
    m = json.loads(json.dumps(meta))
    pts = m.get("points")
    if isinstance(pts, list):
        for p in pts:
            if not isinstance(p, dict):
                continue
            pos = _pos_from_iso(p.get("t"))
            if pos is None:
                return None
            p["i"] = int(pos)
            # Bind timestamp/value to THIS df slice (prevents 'floating' markers on charts)
            try:
                ii = int(p.get("i"))
                if 0 <= ii < len(d):
                    p["t"] = pd.Timestamp(d.index[ii]).isoformat()
                    if "Close" in d.columns:
                        p["p"] = float(d["Close"].iloc[ii])
            except Exception:
                pass
    lns = m.get("lines")
    if isinstance(lns, list):
        for ln in lns:
            if not isinstance(ln, dict):
                continue
            p1 = _pos_from_iso(ln.get("t1"))
            p2 = _pos_from_iso(ln.get("t2"))
            if p1 is None or p2 is None:
                return None
            ln["i1"] = int(p1)
            ln["i2"] = int(p2)
            # Refresh y-values from Close for HS/IHS necklines so they
            # track adjusted prices and stay consistent with point markers.
            if (str(ln.get("label", "")).lower() == "neckline"
                    and str(m.get("annot_type", "")).lower() == "hs"
                    and "Close" in d.columns):
                try:
                    ii1 = int(p1)
                    ii2 = int(p2)
                    if 0 <= ii1 < len(d) and 0 <= ii2 < len(d):
                        ln["y1"] = float(d["Close"].iloc[ii1])
                        ln["y2"] = float(d["Close"].iloc[ii2])
                except Exception:
                    pass
    # pattern start/end from LS/RS points if present
    if isinstance(pts, list):
        ls_i = None
        rs_i = None
        for p in pts:
            if not isinstance(p, dict):
                continue
            lab = str(p.get("label", "")).strip().upper()
            if lab == "LS":
                ls_i = int(p.get("i"))
            if lab == "RS":
                rs_i = int(p.get("i"))
        if ls_i is not None:
            m["pattern_start_i"] = int(ls_i)
        if rs_i is not None:
            m["pattern_end_i"] = int(rs_i)
    return m
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
    # For persistence & freshness: pattern_end is the latest validating touch (not the window end)
    try:
        last_touch_i = int(max((hi_touches or []) + (lo_touches or []))) if (hi_touches or lo_touches) else int(end_i)
    except Exception:
        last_touch_i = int(end_i)
    meta["pattern_end_i"] = int(last_touch_i)
    try:
        meta["pattern_end_t"] = _iso_ts(df.index[int(last_touch_i)])
    except Exception:
        meta["pattern_end_t"] = ""
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
    explain: Optional[Dict[str, int]] = None,
) -> Optional[Tuple[int, int, int, int, int, float, float, float]]:
    """
    Head-first H&S / Inverse H&S selector (deterministic).
    Why this version exists:
    - For HS_TOP we anchor the Head as the global extreme Close in the working window.
    - Shoulders are selected from Close-based local extrema (primary pivots).
      This avoids "intraday High spikes" (e.g., Jan 6) invalidating true Close-shoulders (e.g., Jan 5).
    - We exclude a "head zone" of ±HS_MIN_SIDE_BARS bars around the head from ANY shoulder competition
      (bars right into / out of the head are part of the head run-up, not shoulders).
    - We enforce a minimum reaction depth: between each shoulder and head there must be a valley/peak
      with depth ≥ 2.0 * ATR (config inside this function).
    Returns (p1, p2, p3, t1, t2, px1, px2, px3).
    """
    if len(c) < 10 or d is None or d.empty:
        return None
    def bump(k: str) -> None:
        if isinstance(explain, dict):
            explain[k] = int(explain.get(k, 0)) + 1
    # Primary (shoulder/head) pivots and "between" reaction pivots
    if inverse:
        pivots_primary = [int(x) for x in lows_idx]   # Close-based swing lows
        pivots_between = [int(x) for x in highs_idx]  # OHLC swing highs (reaction highs)
    else:
        pivots_primary = [int(x) for x in highs_idx]  # Close-based swing highs
        pivots_between = [int(x) for x in lows_idx]   # OHLC swing lows (reaction lows)
    if len(pivots_primary) < 2:
        bump("not_enough_primary_pivots")
        return None
    # Head anchor: absolute extreme close in the window (pick MOST RECENT if tied)
    try:
        head_val = float(np.nanmin(c.values)) if inverse else float(np.nanmax(c.values))
        idxs = np.where(np.isclose(c.values.astype(float), head_val, rtol=0.0, atol=1e-8))[0]
        head_target = int(idxs[-1]) if len(idxs) else (int(np.nanargmin(c.values)) if inverse else int(np.nanargmax(c.values)))
    except Exception:
        head_target = None
    if head_target is None:
        bump("no_head")
        return None
    p2 = int(head_target)
    # Candidate shoulder windows around the head (limit search and respect max duration)
    head_zone = int(HS_MIN_SIDE_BARS)  # exclude ± head_zone around head
    L_lo = max(0, p2 - int(HS_MAX_BARS))
    L_hi = max(L_lo, p2 - head_zone)  # exclusive upper
    R_lo = min(len(c), p2 + 1 + head_zone)  # inclusive lower
    R_hi = min(len(c), p2 + 1 + int(HS_MAX_BARS))
    left = [i for i in pivots_primary if L_lo <= i < L_hi]
    right = [i for i in pivots_primary if R_lo <= i < R_hi]
    # Prioritize candidate shoulders by CLOSE (higher first).
    try:
        left = sorted(left, key=lambda i: (float(c.iloc[int(i)]), int(i)), reverse=True)
    except Exception:
        left = list(left)
    try:
        # Right side: close desc, then earlier index first (stability).
        right = sorted(right, key=lambda i: (-float(c.iloc[int(i)]), int(i)))
    except Exception:
        right = list(right)
    if not left or not right:
        bump("no_shoulder_candidates")
        return None
    # Helper: fallback neckline reaction points when swing "between" pivots are missing
    def _safe_nanargmax(v: np.ndarray) -> Optional[int]:
        if v is None or len(v) == 0 or np.all(np.isnan(v)):
            return None
        return int(np.nanargmax(v))
    def _safe_nanargmin(v: np.ndarray) -> Optional[int]:
        if v is None or len(v) == 0 or np.all(np.isnan(v)):
            return None
        return int(np.nanargmin(v))
    def _fallback_t1_t2(p1i: int, p2i: int, p3i: int) -> Optional[Tuple[int, int]]:
        if p2i <= p1i + 1 or p3i <= p2i + 1:
            return None
        if inverse:
            seg1 = d["High"].iloc[p1i + 1 : p2i].to_numpy(dtype=float)
            seg2 = d["High"].iloc[p2i + 1 : p3i].to_numpy(dtype=float)
            j1 = _safe_nanargmax(seg1); j2 = _safe_nanargmax(seg2)
            if j1 is None or j2 is None:
                return None
            return (p1i + 1 + j1, p2i + 1 + j2)
        else:
            seg1 = d["Low"].iloc[p1i + 1 : p2i].to_numpy(dtype=float)
            seg2 = d["Low"].iloc[p2i + 1 : p3i].to_numpy(dtype=float)
            j1 = _safe_nanargmin(seg1); j2 = _safe_nanargmin(seg2)
            if j1 is None or j2 is None:
                return None
            return (p1i + 1 + j1, p2i + 1 + j2)
    # Stronger HS definition: need meaningful pullback between shoulder and head
    REACT_DEPTH_ATR = 2.0
    best: Optional[Tuple[int, int, int, int, int, float, float, float]] = None
    best_score = -1e18
    # Deterministic dominance: a shoulder cannot be "lower" if a higher primary pivot exists
    # in the same shoulder region (excluding the head zone).
    # This prevents selecting Dec 11 when Jan 5 exists, and prevents head-run-up bars invalidating shoulders.
    def _is_dominant_left(p1i: int) -> bool:
        px1 = float(c.iloc[p1i])
        for j in left:
            if j > p1i and j < L_hi:
                if float(c.iloc[j]) > px1 + 1e-9:
                    return False
        return True
    def _is_dominant_right(p3i: int) -> bool:
        px3 = float(c.iloc[p3i])
        for j in right:
            if j < p3i and j >= R_lo:
                if float(c.iloc[j]) > px3 + 1e-9:
                    return False
        return True
    # Iterate candidate shoulder pairs
    for p1 in left:
        p1 = int(p1)
        if p1 >= p2 - head_zone:
            continue
        if not _is_dominant_left(p1):
            bump("ls_not_dominant_left")
            continue
        for p3 in right:
            p3 = int(p3)
            if p3 <= p2 + head_zone:
                continue
            if p3 <= p1:
                continue
            if not _is_dominant_right(p3):
                bump("rs_not_dominant_right")
                continue
            span = int(p3 - p1)
            if span < int(HS_MIN_BARS) or span > int(HS_MAX_BARS):
                bump("duration_or_sidebars")
                continue
            dL = int(p2 - p1)
            dR = int(p3 - p2)
            if dL < int(HS_MIN_SIDE_BARS) or dR < int(HS_MIN_SIDE_BARS):
                bump("duration_or_sidebars")
                continue
            ratio = float(dL) / float(max(1, dR))
            if ratio < 0.33 or ratio > 3.0:
                bump("time_symmetry")
                continue
            px1 = float(c.iloc[p1]); px2 = float(c.iloc[p2]); px3 = float(c.iloc[p3])
            # ATR context
            atr_med = _median_atr(d, p1, p3 + 1)
            price_ref = float(np.nanmedian([px1, px2, px3]))
            min_head_gap = max(0.5 * atr_med, 0.02 * max(price_ref, 1e-6))
            shoulder_tol = max(1.0 * atr_med, 0.05 * max((px1 + px3) / 2.0, 1e-6))
            # Neckline reaction points
            between1 = [x for x in pivots_between if p1 < int(x) < p2]
            between2 = [x for x in pivots_between if p2 < int(x) < p3]
            if between1 and between2:
                if inverse:
                    t1 = max(between1, key=lambda k: float(d["High"].iloc[int(k)]))
                    t2 = max(between2, key=lambda k: float(d["High"].iloc[int(k)]))
                else:
                    t1 = min(between1, key=lambda k: float(d["Low"].iloc[int(k)]))
                    t2 = min(between2, key=lambda k: float(d["Low"].iloc[int(k)]))
            else:
                fb = _fallback_t1_t2(p1, p2, p3)
                if fb is None:
                    bump("fallback_t1t2_failed")
                    continue
                t1, t2 = fb
            # Reaction depth (meaningful pullback into neckline)
            try:
                if inverse:
                    seg1 = d["High"].iloc[p1 + 1 : p2].to_numpy(dtype=float)
                    seg2 = d["High"].iloc[p2 + 1 : p3].to_numpy(dtype=float)
                    if not (np.nanmax(seg1) - px2 >= REACT_DEPTH_ATR * atr_med):
                        bump("reaction_depth")
                        continue
                    if not (np.nanmax(seg2) - px2 >= REACT_DEPTH_ATR * atr_med):
                        bump("reaction_depth")
                        continue
                else:
                    seg1 = d["Low"].iloc[p1 + 1 : p2].to_numpy(dtype=float)
                    seg2 = d["Low"].iloc[p2 + 1 : p3].to_numpy(dtype=float)
                    if not (px2 - np.nanmin(seg1) >= REACT_DEPTH_ATR * atr_med):
                        bump("reaction_depth")
                        continue
                    if not (px2 - np.nanmin(seg2) >= REACT_DEPTH_ATR * atr_med):
                        bump("reaction_depth")
                        continue
            except Exception:
                bump("reaction_depth")
                continue
            # Head/shoulder geometry
            if inverse:
                if not (px2 <= min(px1, px3) - min_head_gap):
                    bump("head_gap")
                    continue
                if abs(px1 - px3) > shoulder_tol:
                    bump("shoulder_mismatch")
                    continue
                head_gap_quality = min(px1 - px2, px3 - px2)
            else:
                if not (px2 >= max(px1, px3) + min_head_gap):
                    bump("head_gap")
                    continue
                if abs(px1 - px3) > shoulder_tol:
                    bump("shoulder_mismatch")
                    continue
                head_gap_quality = min(px2 - px1, px2 - px3)
            # Ensure head is extreme between LS and RS (should be true, but protect against window jitter)
            seg = c.iloc[p1 : p3 + 1]
            if inverse:
                if px2 > float(seg.min()) * (1.0 + 1e-6):
                    bump("head_not_extreme")
                    continue
            else:
                if px2 < float(seg.max()) * (1.0 - 1e-6):
                    bump("head_not_extreme")
                    continue
            # Trend label enforcement
            trend = _trend_context_label(c, int(p2), atr_med)  # anchor on HEAD, not LS
            if inverse and trend != "BOTTOM":
                bump("trend_label")
                continue
            if (not inverse) and trend != "TOP":
                bump("trend_label")
                continue
            # Score
            if inverse:
                n1 = float(d["High"].iloc[int(t1)]); n2 = float(d["High"].iloc[int(t2)])
            else:
                n1 = float(d["Low"].iloc[int(t1)]); n2 = float(d["Low"].iloc[int(t2)])
            sym_penalty = abs(math.log(ratio))
            neck_span = abs(n2 - n1)
            score = float(p3) + 4.0 * (head_gap_quality / max(atr_med, 1e-6)) - 2.0 * sym_penalty - 0.25 * (neck_span / max(atr_med, 1e-6))
            if score > best_score:
                best_score = score
                best = (int(p1), int(p2), int(p3), int(t1), int(t2), float(px1), float(px2), float(px3))
    if best is None:
        bump("no_triplet")
    return best

def _reaction_points_close(
    d: pd.DataFrame,
    p1: int,
    p2: int,
    p3: int,
    inverse: bool,
) -> Tuple[int, int]:
    """Return reaction point indices on Close for HS/IHS.

    HS_TOP (inverse=False): T1/T2 are the closing valleys between LS↔H and H↔RS.
    IHS (inverse=True): R1/R2 are the closing reaction highs between LS↔H and H↔RS.

    We prefer true Close pivots (±3 bars) via swing detection; if none exist, fall back to
    segment argmin/argmax on Close (which will later fail strict local-extreme checks).
    """
    c = d["Close"].astype(float)
    n = len(d)
    p1 = int(max(0, min(p1, n - 1)))
    p2 = int(max(0, min(p2, n - 1)))
    p3 = int(max(0, min(p3, n - 1)))
    if p1 >= p2 or p2 >= p3:
        return (p1, p2)

    if inverse:
        # reaction highs on Close
        piv = _swing_highs_on_close(d, window=HS_LOCAL_WINDOW, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
        left = [i for i in piv if p1 < i < p2]
        right = [i for i in piv if p2 < i < p3]
        if left:
            r1 = max(left, key=lambda k: float(c.iloc[k]))
        else:
            r1 = int(p1 + 1 + int(np.nanargmax(c.iloc[p1 + 1:p2])))
        if right:
            r2 = max(right, key=lambda k: float(c.iloc[k]))
        else:
            r2 = int(p2 + 1 + int(np.nanargmax(c.iloc[p2 + 1:p3])))
        return (int(r1), int(r2))
    else:
        # troughs on Close
        piv = _swing_lows_on_close(d, window=HS_LOCAL_WINDOW, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
        left = [i for i in piv if p1 < i < p2]
        right = [i for i in piv if p2 < i < p3]
        if left:
            t1 = min(left, key=lambda k: float(c.iloc[k]))
        else:
            t1 = int(p1 + 1 + int(np.nanargmin(c.iloc[p1 + 1:p2])))
        if right:
            t2 = min(right, key=lambda k: float(c.iloc[k]))
        else:
            t2 = int(p2 + 1 + int(np.nanargmin(c.iloc[p2 + 1:p3])))
        return (int(t1), int(t2))

def _hs_geometry_diagnostics(
    d: pd.DataFrame,
    p1: int,
    p2: int,
    p3: int,
    inverse: bool = False,
    local_window: int = HS_LOCAL_WINDOW,
    symmetry_min_ratio: float = HS_SYMMETRY_MIN_RATIO,
    valley_atr_mult: float = HS_VALLEY_ATR_MULT,
    shoulder_valley_mult: float = HS_SHOULDER_VALLEY_ATR_MULT,
) -> Dict[str, Any]:
    """Compute deterministic HS/IHS geometry checks on the *same detector window* d.

    Checks (hard fail when any is False):
      1) Head is absolute extreme Close in [LS..RS] (max for HS_TOP, min for IHS), tie-safe.
      2) LS & RS are local extrema on Close within ±local_window bars (max for HS_TOP, min for IHS).
      3) Symmetry ratio min(dL,dR)/max(dL,dR) ≥ symmetry_min_ratio.
      4) Valley depth (Head→Valley) ≥ valley_atr_mult × ATR(head).
      5) Shoulder→Valley depth ≥ shoulder_valley_mult × ATR(head).
      6) Span bars (RS−LS) within [HS_MIN_BARS .. HS_MAX_BARS].
      7) Sidebars: (H−LS) and (RS−H) each ≥ HS_MIN_SIDE_BARS.
    """
    out: Dict[str, Any] = {}
    if d is None or d.empty:
        return {"pass_all": False, "reason": "empty"}

    close = d["Close"].astype(float).values
    high = d["High"].astype(float).values
    low = d["Low"].astype(float).values
    n = len(close)

    p1 = int(p1); p2 = int(p2); p3 = int(p3)
    if not (0 <= p1 < p2 < p3 < n):
        return {"pass_all": False, "reason": "bad_indices"}

    # 6) Span bounds
    span_bars = int(p3 - p1)
    out["span_bars"] = span_bars
    out["span_ok"] = bool((span_bars >= HS_MIN_BARS) and (span_bars <= HS_MAX_BARS))

    # 7) Sidebars
    dL = int(p2 - p1)
    dR = int(p3 - p2)
    out["dL"] = dL
    out["dR"] = dR
    out["sidebars_ok"] = bool((dL >= HS_MIN_SIDE_BARS) and (dR >= HS_MIN_SIDE_BARS))

    # 1) Head must be absolute extreme CLOSE in [LS..RS] span (tie-safe)
    span = close[p1:p3 + 1].astype(float)
    if inverse:
        extreme_val = float(np.nanmin(span))
    else:
        extreme_val = float(np.nanmax(span))
    # prefer the LAST occurrence (tie-safe, consistent with our head anchoring)
    rel_idxs = np.where(np.isclose(span, extreme_val, rtol=0.0, atol=1e-8))[0]
    arg_rel = int(rel_idxs[-1]) if len(rel_idxs) else (int(np.nanargmin(span)) if inverse else int(np.nanargmax(span)))
    head_span_arg = int(p1 + arg_rel)
    head_is_global_span = bool(np.isclose(float(close[p2]), float(extreme_val), rtol=0.0, atol=1e-6))
    out["head_is_global_span"] = head_is_global_span
    out["head_span_arg_i"] = head_span_arg
    out["head_span_extreme_close"] = float(extreme_val)
    out["head_close"] = float(close[p2])

    # 2) LS/RS must be local extrema (Close) within ±local_window bars
    w = int(local_window)

    def _is_local_extreme(i: int) -> bool:
        lo_i = max(0, i - w)
        hi_i = min(n, i + w + 1)
        win = close[lo_i:hi_i]
        if inverse:
            return float(close[i]) <= float(np.nanmin(win) + 1e-8)
        return float(close[i]) >= float(np.nanmax(win) - 1e-8)

    out["ls_local_extreme"] = bool(_is_local_extreme(p1))
    out["rs_local_extreme"] = bool(_is_local_extreme(p3))

    # 3) Symmetry
    ratio = float(min(dL, dR) / max(dL, dR)) if max(dL, dR) > 0 else 0.0
    out["symmetry_ratio"] = ratio
    out["symmetry_ok"] = bool(ratio >= float(symmetry_min_ratio))

    # 4/5) Valley depth vs ATR(head)
    atr_s = atr(d, ATR_N).astype(float).values
    atr_h = float(atr_s[p2]) if np.isfinite(atr_s[p2]) else float(np.nanmedian(atr_s))
    out["atr_head"] = float(atr_h)
    thr = float(valley_atr_mult) * float(atr_h if np.isfinite(atr_h) and atr_h > 0 else 0.0)
    out["valley_thr"] = float(thr)

    if inverse:
        # IHS: valleys are peaks (highs) between LS-H and H-RS
        peakL = float(np.nanmax(high[p1:p2 + 1]))
        peakR = float(np.nanmax(high[p2:p3 + 1]))
        valL = peakL
        valR = peakR
        depthL = peakL - float(close[p2])
        depthR = peakR - float(close[p2])
    else:
        # HS_TOP: valleys are troughs (lows) between LS-H and H-RS
        troughL = float(np.nanmin(low[p1:p2 + 1]))
        troughR = float(np.nanmin(low[p2:p3 + 1]))
        valL = troughL
        valR = troughR
        depthL = float(close[p2]) - troughL
        depthR = float(close[p2]) - troughR

    out["valley_left_depth"] = float(depthL)
    out["valley_right_depth"] = float(depthR)
    out["valley_left_level"] = float(valL)
    out["valley_right_level"] = float(valR)
    out["valley_ok"] = bool((depthL >= thr) and (depthR >= thr))

    shoulder_thr = float(shoulder_valley_mult) * float(atr_h if np.isfinite(atr_h) and atr_h > 0 else 0.0)
    out["shoulder_valley_thr"] = float(shoulder_thr)
    if inverse:
        sdepthL = float(valL) - float(close[p1])
        sdepthR = float(valR) - float(close[p3])
    else:
        sdepthL = float(close[p1]) - float(valL)
        sdepthR = float(close[p3]) - float(valR)
    out["shoulder_valley_left_depth"] = float(sdepthL)
    out["shoulder_valley_right_depth"] = float(sdepthR)
    out["shoulder_valley_ok"] = bool((sdepthL >= shoulder_thr) and (sdepthR >= shoulder_thr))


    # 8) Reaction points (T1/T2 for HS_TOP, R1/R2 for IHS) must be local extrema on Close (±local_window)
    r1, r2 = _reaction_points_close(d, p1, p2, p3, inverse=inverse)
    out["react1_i"] = int(r1)
    out["react2_i"] = int(r2)

    def _is_local_max(i: int) -> bool:
        lo_i = max(0, i - w)
        hi_i = min(n, i + w + 1)
        win = close[lo_i:hi_i]
        return float(close[i]) >= float(np.nanmax(win) - 1e-8)

    def _is_local_min(i: int) -> bool:
        lo_i = max(0, i - w)
        hi_i = min(n, i + w + 1)
        win = close[lo_i:hi_i]
        return float(close[i]) <= float(np.nanmin(win) + 1e-8)

    if inverse:
        out["react_local_ok"] = bool(_is_local_max(r1) and _is_local_max(r2))
    else:
        out["react_local_ok"] = bool(_is_local_min(r1) and _is_local_min(r2))

    # 9) Neckline angle (raw slope) must be within ±HS_NECKLINE_MAX_ANGLE_DEG
    #    IMPORTANT: This is NOT ATR-normalized. It is the geometric slope from the reaction points.
    dc = float(close[r2] - close[r1])   # Δprice
    db = float(r2 - r1)                 # Δbars
    angle_deg = float("nan")
    if db != 0:
        angle_deg = float(math.degrees(math.atan2(dc, db)))
    out["neckline_atr_ref"] = float("nan")  # kept for backward-compatibility of debug dict keys
    out["neckline_angle_deg"] = float(angle_deg)
    out["neckline_angle_ok"] = bool(np.isfinite(angle_deg) and (abs(angle_deg) <= float(HS_NECKLINE_MAX_ANGLE_DEG) + 1e-9))
    out["pass_all"] = bool(
        out["head_is_global_span"]
        and out["ls_local_extreme"]
        and out["rs_local_extreme"]
        and out["symmetry_ok"]
        and out["valley_ok"]
        and out["shoulder_valley_ok"]
        and out["span_ok"]
        and out["sidebars_ok"]
        and out.get("react_local_ok", False)
        and out.get("neckline_angle_ok", False)
    )
    return out
def detect_hs_top(df: pd.DataFrame, explain: Optional[Dict[str, Any]] = None) -> Optional[PatternCandidate]:
    d = df.tail(LOOKBACK_DAYS).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d = _latest_completed_close_df(d)
    if len(d) < 120:
        if isinstance(explain, dict):
            explain['len_lt_120'] = int(explain.get('len_lt_120', 0)) + 1
        return None
    c = d["Close"].astype(float)
    _, lows_idx = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
    highs_idx = _swing_highs_on_close(d, window=5, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
    if len(highs_idx) < 3 or len(lows_idx) < 2:
        if isinstance(explain, dict):
            explain['not_enough_swings'] = int(explain.get('not_enough_swings', 0)) + 1
            explain['highs'] = int(len(highs_idx)); explain['lows'] = int(len(lows_idx))
        return None
    hs = _pick_recent_hs_triplet(highs_idx, lows_idx, c, d, inverse=False, explain=explain)
    if hs is None:
        if isinstance(explain, dict):
            explain['no_triplet'] = int(explain.get('no_triplet', 0)) + 1
        return None
    p1, p2, p3, _t1, _t2, px1, px2, px3 = hs
    t1, t2 = _reaction_points_close(d, p1, p2, p3, inverse=False)
    # --- Geometry guardrails (must pass deterministic criteria) ---
    geom = _hs_geometry_diagnostics(d, p1, p2, p3, inverse=False, local_window=HS_LOCAL_WINDOW,
                                   symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO, valley_atr_mult=HS_VALLEY_ATR_MULT)
    if not geom.get("pass_all", False):
        if isinstance(explain, dict):
            if not geom.get("head_is_global_span", False):
                explain['head_not_global_span'] = int(explain.get('head_not_global_span', 0)) + 1
            if not geom.get("ls_local_extreme", False):
                explain['ls_not_local_extreme'] = int(explain.get('ls_not_local_extreme', 0)) + 1
            if not geom.get("rs_local_extreme", False):
                explain['rs_not_local_extreme'] = int(explain.get('rs_not_local_extreme', 0)) + 1
            if not geom.get("symmetry_ok", False):
                explain['symmetry_fail'] = int(explain.get('symmetry_fail', 0)) + 1
            if not geom.get("valley_ok", False):
                explain['valley_fail'] = int(explain.get('valley_fail', 0)) + 1
            explain['_last_reject_geom'] = {
                'pattern': 'HS_TOP',
                'inverse': False,
                'LS_i': int(p1), 'H_i': int(p2), 'RS_i': int(p3),
                'LS_t': _iso_ts(d.index[int(p1)]), 'H_t': _iso_ts(d.index[int(p2)]), 'RS_t': _iso_ts(d.index[int(p3)]),
                'geom': geom,
            }
        return None
    
    # Neckline rule (deterministic): use the sloped line through (T1,T2) on Close, extended past T2.
    n1 = float(d["Close"].iloc[t1])
    n2 = float(d["Close"].iloc[t2])
    use_horiz = False
    # Fit line y = a*i + b through indices (t1,t2) and extend to last bar.
    if int(t2) != int(t1):
        a_line = (float(n2) - float(n1)) / float(int(t2) - int(t1))
    else:
        a_line = 0.0
    b_line = float(n1) - float(a_line) * float(int(t1))
    neckline_now = float(a_line * float(len(d) - 1) + float(b_line))
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

    # Hard exclusion: do not allow HS if a Dead Cat Bounce overlaps the formation span
    try:
        dcb = detect_dead_cat_bounce(df)
        if dcb and _dcb_overlaps_span(dcb, d.index[p1], d.index[p3]):
            if isinstance(explain, dict):
                explain["dcb_overlap"] = int(explain.get("dcb_overlap", 0)) + 1
            return None
    except Exception:
        pass

    return PatternCandidate(pattern="HS_TOP", direction="BREAKDOWN", level=float(neckline_now), meta=meta)
def detect_inverse_hs(df: pd.DataFrame, explain: Optional[Dict[str, Any]] = None) -> Optional[PatternCandidate]:
    d = df.tail(LOOKBACK_DAYS).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d = _latest_completed_close_df(d)
    if len(d) < 120:
        if isinstance(explain, dict):
            explain['len_lt_120'] = int(explain.get('len_lt_120', 0)) + 1
        return None
    c = d["Close"].astype(float)
    highs_idx, _ = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
    lows_idx = _swing_lows_on_close(d, window=5, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
    if len(lows_idx) < 3 or len(highs_idx) < 2:
        if isinstance(explain, dict):
            explain['not_enough_swings'] = int(explain.get('not_enough_swings', 0)) + 1
            explain['highs'] = int(len(highs_idx)); explain['lows'] = int(len(lows_idx))
        return None
    ihs = _pick_recent_hs_triplet(highs_idx, lows_idx, c, d, inverse=True, explain=explain)
    if ihs is None:
        if isinstance(explain, dict):
            explain['no_triplet'] = int(explain.get('no_triplet', 0)) + 1
        return None
    p1, p2, p3, _r1, _r2, px1, px2, px3 = ihs
    r1, r2 = _reaction_points_close(d, p1, p2, p3, inverse=True)
    # --- Geometry guardrails (must pass deterministic criteria) ---
    geom = _hs_geometry_diagnostics(d, p1, p2, p3, inverse=True, local_window=HS_LOCAL_WINDOW,
                                   symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO, valley_atr_mult=HS_VALLEY_ATR_MULT)
    if not geom.get("pass_all", False):
        if isinstance(explain, dict):
            if not geom.get("head_is_global_span", False):
                explain['head_not_global_span'] = int(explain.get('head_not_global_span', 0)) + 1
            if not geom.get("ls_local_extreme", False):
                explain['ls_not_local_extreme'] = int(explain.get('ls_not_local_extreme', 0)) + 1
            if not geom.get("rs_local_extreme", False):
                explain['rs_not_local_extreme'] = int(explain.get('rs_not_local_extreme', 0)) + 1
            if not geom.get("symmetry_ok", False):
                explain['symmetry_fail'] = int(explain.get('symmetry_fail', 0)) + 1
            if not geom.get("valley_ok", False):
                explain['valley_fail'] = int(explain.get('valley_fail', 0)) + 1
            explain['_last_reject_geom'] = {
                'pattern': 'IHS',
                'inverse': True,
                'LS_i': int(p1), 'H_i': int(p2), 'RS_i': int(p3),
                'LS_t': _iso_ts(d.index[int(p1)]), 'H_t': _iso_ts(d.index[int(p2)]), 'RS_t': _iso_ts(d.index[int(p3)]),
                'geom': geom,
            }
        return None
    h1 = float(d["Close"].iloc[r1]); h2 = float(d["Close"].iloc[r2])  # reaction highs on Close
        # Neckline rule (deterministic): use the sloped line through (R1,R2) on Close, extended past R2.
    use_horiz = False
    if int(r2) != int(r1):
        a_line = (float(h2) - float(h1)) / float(int(r2) - int(r1))
    else:
        a_line = 0.0
    b_line = float(h1) - float(a_line) * float(int(r1))
    neckline_now = float(a_line * float(len(d) - 1) + float(b_line))
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

    # Hard exclusion: do not allow IHS if a Dead Cat Bounce overlaps the formation span
    try:
        dcb = detect_dead_cat_bounce(df)
        if dcb and _dcb_overlaps_span(dcb, d.index[p1], d.index[p3]):
            if isinstance(explain, dict):
                explain["dcb_overlap"] = int(explain.get("dcb_overlap", 0)) + 1
            return None
    except Exception:
        pass
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
    highs_idx, lows_idx = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
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
      - gap-down event: overnight gap-down open >= 10% vs prior close (DCB_MIN_GAP_PCT)
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
    # ATR series used for ATR-based shock gating (e.g., >=5x ATR down day).
    # NOTE: this was previously referenced as `A[...]` without being defined.
    A = atr(d, ATR_N).astype(float).values
    V = pd.to_numeric(d["Volume"], errors="coerce").astype(float).values
    lows_idx_all = _swing_points_ohlc(d, window=3, prominence_atr_mult=0.5)[1]
    best = None
    best_score = -1e18
    # Search recent candidate event days; prioritize recency
    for i in range(max(20, len(d) - 60), len(d) - 8):
        if i < 1:
            continue
        prev_low = float(L[i - 1]); prev_close = float(C[i - 1])
        strict_gap = float(H[i]) < prev_low  # meta only (legacy); not used for gating anymore
        gap_pct = (float(O[i]) / prev_close - 1.0) if prev_close != 0 else 0.0
        day_ret = (float(C[i]) / prev_close - 1.0) if prev_close != 0 else 0.0
        atr_prev = float(A[i - 1]) if np.isfinite(A[i - 1]) else float("nan")
        drop_atr = ((prev_close - float(C[i])) / atr_prev) if (np.isfinite(atr_prev) and atr_prev > 0) else 0.0
        if not ((gap_pct <= -DCB_MIN_GAP_PCT) or (day_ret <= -0.10) or (drop_atr >= 5.0)):
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
        "age_from_event_low_bars": int((len(d) - 1) - best["event_low_i"]),
        "age_from_bounce_high_bars": int((len(d) - 1) - best["bounce_i"]),
    }
    return PatternCandidate(pattern="DEAD_CAT_BOUNCE", direction="BREAKDOWN", level=float(best["trigger"]), meta=meta)
def detect_momo_trend(df: pd.DataFrame) -> Optional[PatternCandidate]:
    """Deterministic 'straight-up' momentum trend detector.
    Why this exists:
      Some names trend relentlessly higher without forming a clean triangle/HS/rectangle.
      We still want them to show up as VALIDATED/CONFIRMED when demand is persistent.
    Definition (all must hold):
      1) Trend: EMA20 > EMA50 and EMA20 rising (EMA20[t] > EMA20[t-5])
      2) Strength: close within 0.25 ATR of the prior 60-day high
      3) Momentum: 20-day return >= 8% OR (close - close_20d_ago) >= 6 * ATR_median
      4) Extension: close >= EMA20 + 0.5 ATR (same confirm distance as other patterns)
    Trigger level used for gating is dynamic EMA20 (meta['dynamic_level']='EMA20').
    Direction: BREAKOUT
    """
    d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 90:
        return None
    # Focus on the recent window for stability.
    look = d.tail(260).copy()
    if len(look) < 90:
        return None
    c = pd.to_numeric(look["Close"], errors="coerce")
    h = pd.to_numeric(look["High"], errors="coerce")
    if c.dropna().shape[0] < 90 or h.dropna().shape[0] < 90:
        return None
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    close_now = float(c.iloc[-1])
    if not np.isfinite(close_now) or close_now <= 0:
        return None
    a = atr(look, n=ATR_N)
    atr_now = float(a.iloc[-1]) if len(a) and np.isfinite(a.iloc[-1]) and float(a.iloc[-1]) > 0 else max(close_now * 0.01, 1e-6)
    a_med = float(pd.to_numeric(a.dropna(), errors="coerce").median()) if not a.dropna().empty else atr_now
    # 1) Trend
    if not (np.isfinite(ema20.iloc[-1]) and np.isfinite(ema50.iloc[-1])):
        return None
    if float(ema20.iloc[-1]) <= float(ema50.iloc[-1]):
        return None
    if len(ema20) < 6 or float(ema20.iloc[-1]) <= float(ema20.iloc[-6]):
        return None
    # 2) Strength vs prior 60-day high (exclude the current bar)
    if len(h) < 61:
        return None
    prior60_high = float(h.rolling(60).max().shift(1).iloc[-1])
    if not np.isfinite(prior60_high) or prior60_high <= 0:
        return None
    if close_now < (prior60_high - 0.25 * atr_now):
        return None
    # 3) Momentum
    if len(c) < 21 or not np.isfinite(c.iloc[-21]) or float(c.iloc[-21]) <= 0:
        return None
    ret20 = (close_now / float(c.iloc[-21]) - 1.0)
    if not (ret20 >= 0.08 or (close_now - float(c.iloc[-21])) >= 6.0 * a_med):
        return None
    # 4) Extension above EMA20 (keeps it "on fire" rather than a gentle drift)
    if close_now < float(ema20.iloc[-1]) + 0.5 * atr_now:
        return None
    meta: Dict[str, Any] = {
        "annot_type": "momo",
        "dynamic_level": "EMA20",
        "prior60_high": prior60_high,
        "ret20_pct": float(ret20 * 100.0),
    }
    # cand.level is a placeholder; gating uses the dynamic EMA20 via _level_at_bar.
    return PatternCandidate(pattern="MOMO_TREND", direction="BREAKOUT", level=float(ema20.iloc[-1]), meta=meta)
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
    # Geometry-based band structures (triangles / broadening)
    out.extend(detect_structure_candidates(df))
    # Momentum trend (straight-up) — deterministic, for names that trend without clean geometry
    momo = detect_momo_trend(df)
    if momo:
        out.append(momo)
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
    Returns (prefix, distance_in_atr), where prefix in {"", "EARLY_", "CONFIRMED_"}.
    """
    if atr_val is None or atr_val <= 0 or math.isnan(atr_val):
        base = abs(level) if level is not None else abs(close)
        atr_val = max(base * 0.01, 1e-6)
    # Normalize possibly-missing inputs
    try:
        if vol_ratio is None or (isinstance(vol_ratio, float) and math.isnan(vol_ratio)):
            vol_ratio = 1.0
    except Exception:
        vol_ratio = 1.0
    try:
        if clv is None or (isinstance(clv, float) and math.isnan(clv)):
            clv = 0.0
    except Exception:
        clv = 0.0
    dist_atr = (float(close) - float(level)) / float(atr_val)
    if str(direction).upper() == "BREAKOUT":
        price_ok = float(close) >= float(level) + ATR_CONFIRM_MULT * float(atr_val)
        vol_ok = float(vol_ratio) >= VOL_CONFIRM_MULT
        clv_ok = float(clv) >= CLV_BREAKOUT_MIN
        if price_ok:
            return "CONFIRMED_", float(dist_atr)
        if abs(float(close) - float(level)) <= EARLY_MULT * float(atr_val):
            return "EARLY_", float(dist_atr)
        return "", float(dist_atr)
    # BREAKDOWN (default branch for any non-BREAKOUT direction)
    price_ok = float(close) <= float(level) - ATR_CONFIRM_MULT * float(atr_val)
    vol_ok = float(vol_ratio) >= VOL_CONFIRM_MULT
    clv_ok = float(clv) <= CLV_BREAKDOWN_MAX
    if price_ok:
        return "CONFIRMED_", float(dist_atr)
    if abs(float(close) - float(level)) <= EARLY_MULT * float(atr_val):
        return "EARLY_", float(dist_atr)
    return "", float(dist_atr)
def _bar_clv(d: pd.DataFrame, i: int) -> float:
    try:
        close = float(d["Close"].iloc[i])
        hi = float(d["High"].iloc[i])
        lo = float(d["Low"].iloc[i])
        if hi > lo:
            v = (2.0 * close - hi - lo) / (hi - lo)
            return float(max(-1.0, min(1.0, v)))
    except Exception:
        pass
    return 0.0
def _bar_vol_ratio(d: pd.DataFrame, i: int) -> float:
    # Volume ratio vs prior 20 sessions (exclude current bar i)
    try:
        if "Volume" not in d.columns:
            return 1.0
        v = float(d["Volume"].iloc[i])
        if not np.isfinite(v):
            return 1.0
        start = max(0, i - 20)
        end = i
        if end - start < 5:
            return 1.0
        avg = float(pd.to_numeric(d["Volume"].iloc[start:end], errors="coerce").dropna().mean())
        if np.isfinite(avg) and avg > 0:
            return float(v / avg)
    except Exception:
        pass
    return 1.0
def _level_at_bar(cand: PatternCandidate, d: pd.DataFrame, i: int) -> float:
    # Default to static level
    lvl = float(cand.level)
    meta = cand.meta if isinstance(cand.meta, dict) else {}
    # Dynamic trigger levels (used for MOMO_TREND and future indicators).
    # When set, we ignore cand.level and compute the level from the OHLCV series.
    try:
        dyn = str(meta.get("dynamic_level", "")).strip().lower()
    except Exception:
        dyn = ""
    if dyn == "ema20":
        try:
            ema20 = d["Close"].astype(float).ewm(span=20, adjust=False).mean()
            if 0 <= int(i) < len(ema20) and np.isfinite(ema20.iloc[int(i)]):
                return float(ema20.iloc[int(i)])
        except Exception:
            pass
    lines = meta.get("lines") if isinstance(meta, dict) else None
    if not isinstance(lines, list) or not lines:
        return lvl
    want = None
    if cand.pattern in ("HS_TOP", "IHS"):
        want = "Neckline"
    elif cand.pattern == "DEAD_CAT_BOUNCE":
        want = "Active trigger"
    else:
        # Band patterns
        if cand.direction == "BREAKOUT":
            want = "Upper"
        else:
            want = "Lower"
    chosen = None
    for ln in lines:
        if isinstance(ln, dict) and str(ln.get("label", "")).lower() == str(want).lower():
            chosen = ln
            break
    if chosen is None and isinstance(lines[0], dict):
        chosen = lines[0]
    try:
        i1 = int(chosen.get("i1"))
        i2 = int(chosen.get("i2"))
        y1 = float(chosen.get("y1"))
        y2 = float(chosen.get("y2"))
        a, b = _line_fit(np.array([float(i1), float(i2)]), np.array([float(y1), float(y2)]))
        return float(_line_eval(a, b, float(i)))
    except Exception:
        return lvl
def _is_confirmed_bar(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    i: int,
    atr_mult: float = ATR_CONFIRM_MULT,
) -> bool:
    """Return True if bar i satisfies the 3 hard confirmation gates."""
    try:
        i = int(i)
        if i < 0 or i >= len(d):
            return False
        level = _safe_float(_level_at_bar(cand, d, i))
        close = _safe_float(d["Close"].iloc[i])
        if math.isnan(level) or math.isnan(close):
            return False
        atr_v = _safe_float(a_series.iloc[i]) if a_series is not None and len(a_series) > i else float("nan")
        if math.isnan(atr_v) or atr_v <= 0:
            return False
        dist = (close - level) / atr_v
        if cand.direction == "BREAKOUT":
            if dist < atr_mult:
                return False
        else:
            if dist > -atr_mult:
                return False
        clv = _clv_at_bar(d, i)
        if cand.direction == "BREAKOUT":
            if clv < CLV_BREAKOUT_MIN:
                return False
        else:
            if clv > CLV_BREAKDOWN_MAX:
                return False
        if "Volume" not in d.columns:
            return False
        v = _safe_float(d["Volume"].iloc[i])
        if math.isnan(v) or v <= 0:
            return False
        if i >= 21:
            avg20 = float(pd.to_numeric(d["Volume"].iloc[i-21:i-1], errors="coerce").mean())
        else:
            avg20 = float(pd.to_numeric(d["Volume"].iloc[:i], errors="coerce").tail(20).mean()) if i > 1 else float("nan")
        if math.isnan(avg20) or avg20 <= 0:
            return False
        if v < VOL_CONFIRM_MULT * avg20:
            return False
        return True
    except Exception:
        return False
def _is_price_ok_bar(cand: PatternCandidate, d: pd.DataFrame, a_series: pd.Series, i: int) -> bool:
    """Price-only gate: close beyond trigger by >= ATR_CONFIRM_MULT * ATR (directional)."""
    close_i = _safe_float(d["Close"].iloc[i])
    atr_i = _safe_float(a_series.iloc[i]) if i < len(a_series) else float('nan')
    if not np.isfinite(atr_i) or atr_i <= 0:
        atr_i = max(close_i * 0.01, 1e-6)
    level_i = _safe_float(_level_at_bar(cand, d, i))
    if cand.direction == "BREAKOUT":
        return bool(close_i >= level_i + ATR_CONFIRM_MULT * atr_i)
    return bool(close_i <= level_i - ATR_CONFIRM_MULT * atr_i)
def _validated_run_start_after_last_failure(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    end_idx: int,
) -> Optional[int]:
    """Find breakout/breakdown start for VALIDATED lifecycle.
    - Find last bar where price was on the wrong side of the trigger (close < level for breakout; close > level for breakdown).
    - After that, take the first bar that satisfies the price-only confirm gate (>= 0.5 ATR beyond trigger).
    This avoids requiring elevated Volume/CLV on *every* subsequent day.
    """
    n = len(d)
    if n < 10:
        return None
    end_idx = int(min(max(end_idx, 0), n - 1))
    lookback = int(min(n - 1, VALIDATED_MAX_AGE_BARS + 60))
    start_scan = max(0, end_idx - lookback)
    last_wrong = None
    for i in range(end_idx, start_scan - 1, -1):
        close_i = _safe_float(d["Close"].iloc[i])
        level_i = _safe_float(_level_at_bar(cand, d, i))
        if cand.direction == "BREAKOUT":
            if close_i < level_i:
                last_wrong = i
                break
        else:
            if close_i > level_i:
                last_wrong = i
                break
    start_search = (last_wrong + 1) if last_wrong is not None else start_scan
    for i in range(start_search, end_idx + 1):
        try:
            if _is_price_ok_bar(cand, d, a_series, i):
                return int(i)
        except Exception:
            continue
    return None
def _validation_window_ok(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    run_start: int,
) -> bool:
    """3-day deterministic HS lifecycle validation (as agreed).

    Definitions (relative to run_start = confirmation day):
      - Day 0 (run_start): ALL 3 gates must hold (price + CLV + volume).
      - Day 1 (run_start+1): keep the run alive if price + volume hold; CLV is optional.
      - Day 2 (run_start+2): VALIDATED requires ALL 3 gates again.

    So the 3-bar window (day0..day2) must satisfy:
      - day0: confirmed (3 gates)
      - day1: price+volume (CLV ignored)
      - day2: confirmed (3 gates)
    """
    n = len(d)
    rs = int(run_start)
    if rs < 0 or rs + 2 >= n:
        return False
    # day0: strict confirmed
    if not _is_confirmed_bar(cand, d, a_series, rs, atr_mult=ATR_CONFIRM_MULT):
        return False
    # day1: price+volume (CLV optional)
    if not _is_pricevol_bar(cand, d, a_series, rs + 1, atr_mult=ATR_CONFIRM_MULT):
        return False
    # day2: strict confirmed again
    if not _is_confirmed_bar(cand, d, a_series, rs + 2, atr_mult=ATR_CONFIRM_MULT):
        return False
    return True

def _validated_stage(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    end_idx: int,
) -> Optional[Tuple[str, str, int, int]]:
    """Return (stage,status,age,run_start) for VALIDATED if applicable, else None.

    Deterministic & time-bounded validation (3-day lifecycle):
      - Find the most recent run start rs such that the 3-bar window rs..rs+2 satisfies:
          day0 (rs): all 3 gates (price + CLV + volume)
          day1 (rs+1): price + volume (CLV optional)
          day2 (rs+2): all 3 gates again
      - VALIDATED_NEW: age == 2 (today is day2)
      - VALIDATED_ONGOING: age > 2 up to VALIDATED_MAX_AGE_BARS,
        provided price remains on the correct side of the trigger.
      - If a run never validates by day2, it expires (not shown).
    """
    n = len(d)
    if n < VALIDATE_BARS + 2:
        return None
    end_idx = int(min(max(end_idx, 0), n - 1))
    # scan window: last VALIDATED_MAX_AGE_BARS + buffer
    max_scan_back = int(min(n - 1, VALIDATED_MAX_AGE_BARS + VALIDATE_BARS + 20))
    start_scan = max(0, end_idx - max_scan_back)
    rs_found = None
    for rs in range(end_idx - VALIDATED_MIN_AGE_BARS, start_scan - 1, -1):
        rs = int(rs)
        if rs < 0 or rs + VALIDATE_BARS - 1 >= n:
            continue
        # must be a run start (previous bar not confirmed)
        if rs - 1 >= 0 and _is_confirmed_bar(cand, d, a_series, rs - 1):
            continue
        if _validation_window_ok(cand, d, a_series, rs):
            rs_found = rs
            break
    if rs_found is None:
        return None
    age = int(end_idx - int(rs_found))
    if age < VALIDATED_MIN_AGE_BARS or age > VALIDATED_MAX_AGE_BARS:
        return None
    # ongoing validity: still beyond trigger
    close_now = _safe_float(d["Close"].iloc[end_idx])
    level_now = _safe_float(_level_at_bar(cand, d, end_idx))
    if cand.direction == "BREAKOUT" and close_now < level_now:
        return None
    if cand.direction != "BREAKOUT" and close_now > level_now:
        return None
    status = "NEW" if age == VALIDATED_MIN_AGE_BARS else "ONGOING"
    return ("VALIDATED", status, age, int(rs_found))
def _is_pricevol_bar(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    i: int,
    atr_mult: float = ATR_CONFIRM_MULT,
) -> bool:
    """Return True if bar i satisfies the price+volume gates (CLV ignored).
    Used to keep CONFIRMED signals alive on day+1 even if CLV is noisy.
    """
    try:
        i = int(i)
        if i < 0 or i >= len(d):
            return False
        level = _safe_float(_level_at_bar(cand, d, i))
        close = _safe_float(d["Close"].iloc[i])
        if math.isnan(level) or math.isnan(close):
            return False
        atr_v = _safe_float(a_series.iloc[i]) if a_series is not None and len(a_series) > i else float("nan")
        if math.isnan(atr_v) or atr_v <= 0:
            return False
        dist = (close - level) / atr_v
        if cand.direction == "BREAKOUT":
            if dist < atr_mult:
                return False
        else:
            if dist > -atr_mult:
                return False
        if "Volume" not in d.columns:
            return False
        v = _safe_float(d["Volume"].iloc[i])
        if math.isnan(v) or v <= 0:
            return False
        if i >= 21:
            avg20 = float(pd.to_numeric(d["Volume"].iloc[i-21:i-1], errors="coerce").mean())
        else:
            avg20 = float(pd.to_numeric(d["Volume"].iloc[:i], errors="coerce").tail(20).mean()) if i > 1 else float("nan")
        if math.isnan(avg20) or avg20 <= 0:
            return False
        if v < VOL_CONFIRM_MULT * avg20:
            return False
        return True
    except Exception:
        return False

# ----------------------------
# Exit rules (applied to all formations once CONFIRMED/VALIDATED)
# ----------------------------
def _atr_ref_for_exit(cand: PatternCandidate, d: pd.DataFrame, a_series: pd.Series, run_start: int) -> float:
    """ATR reference used for exit trailing.
    - HS/IHS: prefer ATR at Head (structure anchor)
    - Otherwise: ATR at run_start
    Falls back to median ATR, then ~1% of price.
    """
    atr_ref = float("nan")
    try:
        rs = int(run_start)
        if a_series is not None and 0 <= rs < len(a_series):
            atr_ref = float(a_series.iloc[rs])
    except Exception:
        atr_ref = float("nan")
    # Prefer head ATR for HS/IHS if available
    try:
        if isinstance(getattr(cand, "meta", None), dict) and str(cand.meta.get("annot_type", "")) == "hs":
            tri = _hs_indices_from_meta(cand.meta)
            if tri is not None:
                _ls_i, h_i, _rs_i = tri
                if a_series is not None and 0 <= int(h_i) < len(a_series):
                    v = float(a_series.iloc[int(h_i)])
                    if np.isfinite(v) and v > 0:
                        atr_ref = v
    except Exception:
        pass
    try:
        if (not np.isfinite(atr_ref)) or atr_ref <= 0:
            atr_ref = float(np.nanmedian(np.asarray(a_series, dtype=float)))
    except Exception:
        pass
    if (not np.isfinite(atr_ref)) or atr_ref <= 0:
        try:
            px = float(d["Close"].iloc[int(run_start)])
            atr_ref = max(px * 0.01, 1e-6)
        except Exception:
            atr_ref = 1e-6
    return float(atr_ref)

def _exit_check_giveback(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    run_start: int,
    end_idx: int,
    giveback_atr: float = EXIT_GIVEBACK_ATR,
) -> Dict[str, Any]:
    """Trailing exit: allow at most `giveback_atr` ATR_ref giveback from peak excursion.

    Define excursion/excess in ATR units vs the trigger/neckline level at each bar:
      - BREAKOUT: excess = (Close - Level)/ATR_ref
      - BREAKDOWN: excess = (Level - Close)/ATR_ref

    Track peak_excess since run_start.
    Exit if:
      - excess <= 0 (back through trigger), OR
      - peak_excess >= giveback_atr and excess <= peak_excess - giveback_atr
    """
    out: Dict[str, Any] = {"exit": False, "reason": None}
    if not EXIT_ENABLE:
        return out
    n = len(d)
    rs = int(max(0, min(int(run_start), n - 1)))
    ei = int(max(0, min(int(end_idx), n - 1)))
    if ei <= rs:
        return out
    atr_ref = _atr_ref_for_exit(cand, d, a_series, rs)
    if not np.isfinite(atr_ref) or atr_ref <= 0:
        return out
    excess_vals: List[float] = []
    for j in range(rs, ei + 1):
        try:
            level_j = _safe_float(_level_at_bar(cand, d, j))
            close_j = _safe_float(d["Close"].iloc[j])
            if not np.isfinite(level_j) or not np.isfinite(close_j):
                continue
            if cand.direction == "BREAKOUT":
                ex = (close_j - level_j) / atr_ref
            else:
                ex = (level_j - close_j) / atr_ref
            excess_vals.append(float(ex))
        except Exception:
            continue
    if not excess_vals:
        return out
    peak_excess = float(np.nanmax(np.asarray(excess_vals, dtype=float)))
    cur_excess = float(excess_vals[-1])
    out.update({"atr_ref": float(atr_ref), "peak_excess": peak_excess, "cur_excess": cur_excess, "giveback_atr": float(giveback_atr)})
    # Hard invalidation: back through trigger/neckline
    if cur_excess <= 0.0:
        out["exit"] = True
        out["reason"] = "EXIT_NECKLINE"
        return out
    # Trailing giveback from max favorable excursion
    if peak_excess >= float(giveback_atr) and cur_excess <= (peak_excess - float(giveback_atr)):
        out["exit"] = True
        out["reason"] = "EXIT_GIVEBACK"
        return out
    return out



def _historical_confirm_then_retest(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a: pd.Series,
    start_i: int,
) -> bool:
    """
    If the pattern already had a *break attempt* in the past (price gate hit on some bar >= start_i),
    and price has since returned to/below the trigger (excess<=0), treat a current near-trigger setup
    as a failed retest (do NOT re-flag as EARLY).

    Rationale: we don't want a geometry to keep resurfacing as EARLY after it already broke out/broke down
    and then reverted back to the neckline. This scan is limited to the detector window (LOOKBACK_DAYS),
    so it won't suppress unrelated formations years later.
    """
    try:
        n = len(d)
        if n < 10:
            return False
        si = int(max(0, min(int(start_i), n - 1)))
        attempt_i = None
        for j in range(si, n):
            g = _gates_for_signal(cand, d, a, j)
            # Price gate implies a decisive move beyond the confirmation band (±0.5 ATR).
            if g.get("price"):
                attempt_i = j
                break
        if confirmed_i is None:
            return False
        atr_ref = _atr_ref_for_exit(cand, d, a, confirmed_i)
        if not np.isfinite(atr_ref) or atr_ref <= 0:
            atr_ref = float(a.iloc[confirmed_i]) if np.isfinite(a.iloc[confirmed_i]) else 1e-6
        if atr_ref <= 0:
            return False
        for j in range(confirmed_i, n):
            lvl = _safe_float(_level_at_bar(cand, d, j))
            close = _safe_float(d["Close"].iloc[j])
            if not (np.isfinite(lvl) and np.isfinite(close)):
                continue
            excess = (close - lvl) / atr_ref if cand.direction == "BREAKOUT" else (lvl - close) / atr_ref
            if excess <= 0:
                return True
        return False
    except Exception:
        return False

def _historical_big_pierce_then_revert(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a: pd.Series,
    start_i: int,
    pierce_mult: float = 2.0,
) -> bool:
    """
    EARLY exit (close-based blow-off failure) — deterministic.

    Trigger an EARLY exit if, after start_i:
      - BREAKOUT: Close >= level + pierce_mult * ATR(14)  AND the bar is NOT confirmed (fails CLV and/or Volume gate)
      - BREAKDOWN: Close <= level - pierce_mult * ATR(14) AND the bar is NOT confirmed

    No intraday High/Low logic (too noisy). No "later revert" requirement: the moment we close that far beyond
    the level without confirmation, we treat it as a failed/unstable attempt and exit EARLY.
    """
    try:
        n = len(d)
        if n < 10:
            return False
        si = int(max(0, min(int(start_i), n - 2)))

        def _atr_at(j: int) -> float:
            try:
                v = float(a.iloc[j])
                return v if np.isfinite(v) and v > 0 else 0.0
            except Exception:
                return 0.0

        def _cl_at(j: int) -> float:
            try:
                return float(d["Close"].iloc[j])
            except Exception:
                return float("nan")

        for j in range(si, n):
            lvl = _safe_float(_level_at_bar(cand, d, j))
            atr_j = _atr_at(j)
            cl = _cl_at(j)
            if not (np.isfinite(lvl) and np.isfinite(atr_j) and atr_j > 0 and np.isfinite(cl)):
                continue

            # Close-based pierce
            if cand.direction == "BREAKOUT":
                if cl >= float(lvl) + float(pierce_mult) * float(atr_j):
                    # If it is confirmed, we don't treat it as a blow-off failure.
                    if not _is_confirmed_bar(cand, d, a_series=a, i=j, atr_mult=ATR_CONFIRM_MULT):
                        return True
            else:
                if cl <= float(lvl) - float(pierce_mult) * float(atr_j):
                    if not _is_confirmed_bar(cand, d, a_series=a, i=j, atr_mult=ATR_CONFIRM_MULT):
                        return True

        return False
    except Exception:
        return False
def _confirm_run_start(cand: PatternCandidate, d: pd.DataFrame, a_series: pd.Series) -> Optional[int]:
    """Return the index (in d) of the first bar of the current CONFIRMED run, or None.
    Day 0 (confirmation day): ALL 3 gates must hold (price + CLV + volume).
    Day 1 (the next session): keep CONFIRMED alive as ONGOING if price + volume still hold; CLV is optional.
    (Validated still requires all 3 gates on each session per the validated-stage rules.)
    This is used to deterministically label signals as NEW/ONGOING and to transition to VALIDATED.
    """
    n = len(d)
    if n < 5:
        return None
    end = n - 1
    # Case A: fully confirmed today -> walk back contiguous fully-confirmed bars
    if _is_confirmed_bar(cand, d, a_series, end):
        j = end
        while j > 0 and _is_confirmed_bar(cand, d, a_series, j - 1):
            j -= 1
        return int(j)
    # Case B: day+1 carry (CLV optional) -> yesterday was fully confirmed, today still meets price+volume
    if end >= 1:
        if _is_pricevol_bar(cand, d, a_series, end) and _is_confirmed_bar(cand, d, a_series, end - 1):
            # Keep as ONGOING confirmed for one extra day; run starts yesterday so age==1.
            return int(end - 1)
    return None
def _stage_from_confirm_run(
    cand: PatternCandidate,
    d: pd.DataFrame,
    a_series: pd.Series,
    run_start: int,
) -> Tuple[str, str, int]:
    """Deterministically classify a signal after a CONFIRMED run is present.

    Lifecycle (relative to run_start = Day 0):
      - Day 0: CONFIRMED requires all 3 gates.
      - Day 1: CONFIRMED_ONGOING requires price+volume (CLV optional).
      - Day 2: VALIDATED requires all 3 gates again (and day1 price+volume).
      - After Day 2, VALIDATED_ONGOING remains while price stays beyond the trigger (capped).
    """
    n = len(d)
    age = int((n - 1) - int(run_start))
    # If the run is old enough to validate, it must have validated exactly on day 3 or it expires.
    if age >= VALIDATED_MIN_AGE_BARS:
        ok = True
        for k in range(int(run_start), int(run_start) + VALIDATED_MIN_AGE_BARS + 1):
            if k >= n or not _is_confirmed_bar(cand, d, a_series, k, atr_mult=ATR_CONFIRM_MULT):
                ok = False
                break
        if not ok:
            return ("EXPIRED", "EXPIRED", age)
        # Cap how long we keep VALIDATED ongoing (config knob)
        if age > VALIDATED_MAX_AGE_BARS:
            return ("EXPIRED", "EXPIRED", age)
        status = "NEW" if age == VALIDATED_MIN_AGE_BARS else "ONGOING"
        return ("VALIDATED", status, age)
    # Otherwise still in the short CONFIRMED window (0..2)
    if age > CONFIRMED_MAX_AGE_BARS:
        return ("EXPIRED", "EXPIRED", age)
    status = "NEW" if age == 0 else "ONGOING"
    return ("CONFIRMED", status, age)
def compute_signals_for_ticker(ticker: str, df: pd.DataFrame, state: Optional[Dict[str, Any]] = None, debug: Optional[Dict[str, Any]] = None) -> List[LevelSignal]:
    sigs: List[LevelSignal] = []
    if df is None or df.empty or len(df) < 80:
        return sigs
    # IMPORTANT: use the same lookback slice for detection + level evaluation so meta indices stay aligned.
    d0 = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(d0) < 80:
        return sigs
    # Detection lookback: ALWAYS use tail(LOOKBACK_DAYS) so HS/IHS meta indices (iloc) align everywhere.
    d = d0.tail(LOOKBACK_DAYS).copy()
    d = _latest_completed_close_df(d)
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
    # Debug: candidate counts
    if isinstance(debug, dict):
        debug["cand_total"] = int(debug.get("cand_total", 0)) + int(len(candidates))
        byp = debug.setdefault("cand_by_pattern", {})
        for cnd in candidates:
            k = str(getattr(cnd, "pattern", ""))
            byp[k] = int(byp.get(k, 0)) + 1
    # HS/IHS geometry carry-forward: survive pivot re-picks on big bars.
    if isinstance(state, dict):
        hs_geom = state.setdefault("hs_geom", {})
        mem = hs_geom.get(ticker)
        have_hs_today = any(getattr(cnd, "pattern", "") in ("HS_TOP", "IHS") for cnd in candidates)
        if (not have_hs_today) and isinstance(mem, dict):
            try:
                asof = mem.get("asof")
                age_ok = True
                if isinstance(d.index, pd.DatetimeIndex) and asof is not None:
                    asof_dt = pd.to_datetime(str(asof), utc=True, errors="coerce")
                    if pd.isna(asof_dt):
                        asof_dt = pd.to_datetime(str(asof), errors="coerce")
                    if not pd.isna(asof_dt):
                        asof_key = asof_dt.date().isoformat()
                        date_keys = [pd.Timestamp(x).date().isoformat() for x in d.index]
                        if asof_key in date_keys:
                            age = int(len(date_keys) - 1 - date_keys.index(asof_key))
                            age_ok = age <= HS_GEOM_CARRY_BARS
                        else:
                            age_ok = False
                if age_ok:
                    meta2 = _reindex_meta_to_df(mem.get("meta", {}), d)
                    # Guardrail: if we carry HS/IHS geometry forward, it must still pass deterministic checks.
                    if isinstance(meta2, dict) and str(meta2.get("annot_type","")) == "hs":
                        try:
                            pts = {str(p.get("label","")): p for p in (meta2.get("points") or []) if isinstance(p, dict)}
                            pLS = pts.get("LS"); pH = pts.get("H"); pRS = pts.get("RS")
                            if pLS and pH and pRS:
                                ls_i = int(pLS.get("i")); h_i = int(pH.get("i")); rs_i = int(pRS.get("i"))
                                inv = (str(mem.get("pattern","")) == "IHS") or (str(meta2.get("variant","")) == "inverse")
                                g = _hs_geometry_diagnostics(d, ls_i, h_i, rs_i, inverse=inv,
                                                             local_window=HS_LOCAL_WINDOW,
                                                             symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO,
                                                             valley_atr_mult=HS_VALLEY_ATR_MULT)
                                if not g.get("pass_all", False):
                                    meta2 = None
                        except Exception:
                            pass
                    if meta2 is not None:
                        candidates.append(PatternCandidate(
                            pattern=str(mem.get("pattern", "")),
                            direction=str(mem.get("direction", "")),
                            level=float(mem.get("level", 0.0)),
                            meta=meta2,
                        ))
                        if isinstance(debug, dict):
                            debug['hs_restored'] = int(debug.get('hs_restored', 0)) + 1
                else:
                    hs_geom.pop(ticker, None)
            except Exception:
                pass
        # update memory if HS/IHS candidate exists today
        try:
            best = next(cnd for cnd in candidates if getattr(cnd, "pattern", "") in ("HS_TOP", "IHS"))
            hs_geom[ticker] = {
                "pattern": best.pattern,
                "direction": best.direction,
                "level": float(best.level),
                "meta": best.meta,
                "asof": _iso_ts(d.index[-1]),
            }
        except Exception:
            pass
    # Band geometry carry-forward: rectangles/triangles/broadening can flip-flop due to refits.
    # Persist neutral geometry (upper+lower) keyed by last validating touch, for up to 30 bars.
    if isinstance(state, dict):
        band_geom = state.setdefault("band_geom", {})
        mem_b = band_geom.get(ticker)
        have_band_today = any(isinstance(getattr(cnd, "meta", None), dict) and str(cnd.meta.get("annot_type", "")) == "band" for cnd in candidates)
        if (not have_band_today) and isinstance(mem_b, dict):
            try:
                last_touch = str(mem_b.get("last_touch", "") or "")
                if isinstance(d.index, pd.DatetimeIndex) and last_touch:
                    date_keys = [pd.Timestamp(x).date().isoformat() for x in d.index]
                    lt_dt = pd.to_datetime(last_touch, utc=True, errors="coerce")
                    if pd.isna(lt_dt):
                        lt_dt = pd.to_datetime(last_touch, errors="coerce")
                    lt_key = lt_dt.date().isoformat() if not pd.isna(lt_dt) else ""
                    if lt_key in date_keys:
                        age = int(len(date_keys) - 1 - date_keys.index(lt_key))
                        if age <= BAND_GEOM_CARRY_BARS:
                            meta2 = _reindex_meta_to_df(mem_b.get("meta", {}), d)
                            if meta2 is not None:
                                pat = str(mem_b.get("pattern", ""))
                                candidates.append(PatternCandidate(pattern=pat, direction="BREAKOUT", level=0.0, meta=meta2))
                                candidates.append(PatternCandidate(pattern=pat, direction="BREAKDOWN", level=0.0, meta=meta2))
                                if isinstance(debug, dict):
                                    debug["band_restored"] = int(debug.get("band_restored", 0)) + 1
                    else:
                        band_geom.pop(ticker, None)
            except Exception:
                pass
        # Update memory if a band candidate exists today
        try:
            best_band = next(cnd for cnd in candidates if isinstance(getattr(cnd, "meta", None), dict) and str(cnd.meta.get("annot_type", "")) == "band")
            meta_b = best_band.meta or {}
            last_touch_t = str(meta_b.get("pattern_end_t", "") or "")
            band_geom[ticker] = {
                "pattern": str(best_band.pattern),
                "meta": meta_b,
                "last_touch": last_touch_t,
            }
        except Exception:
            pass
    # De-duplicate candidates (same pattern/dir/trigger rounded)
    seen = set()
    for cand in candidates:
        key = (cand.pattern, cand.direction, round(float(cand.level), 4))
        if key in seen:
            continue
        seen.add(key)
        # Stage logic (deterministic lifecycle):
        # - EARLY: within 0.5 ATR of trigger (pre-break), regardless of volume/CLV gates.
        # - CONFIRMED (Day 0): breakout/breakdown day where ALL 3 gates hold
        #   (price beyond trigger by >=0.5 ATR, CLV >=+0.70 / <=-0.70, Vol >=1.25x AvgVol20).
        # - CONFIRMED_ONGOING (Day 1): keep alive if price + volume hold; CLV is optional.
        # - VALIDATED (Day 2): requires ALL 3 gates again (day0 all3, day1 price+vol, day2 all3).
        #   After that it remains VALIDATED_ONGOING (capped by VALIDATED_MAX_AGE_BARS) while price stays
        #   on the correct side of the trigger; otherwise it expires.
        curr_level = _level_at_bar(cand, d, len(d) - 1)
        level_now = float(curr_level)
        dist_atr = (close - level_now) / (atr_val if np.isfinite(atr_val) and atr_val > 0 else max(float(abs(level_now)) * 0.01, 1e-6))
        vp_runway_pct = None
        vp_zone_low = None
        vp_zone_high = None
        stage_status = None
        stage_age_bars = None
        breakout_start = None
        run_start = _confirm_run_start(cand, d, a)  # (kept for CONFIRMED-only)
        # Prefer VALIDATED lifecycle (can remain active even if today is not "fully confirmed")
        vinfo = _validated_stage(cand, d, a, len(d) - 1)
        if vinfo is not None:
            stage, status, age, rs = vinfo
            prefix = f"{stage}_"
            stage_status = status
            stage_age_bars = int(age)
            try:
                breakout_start = str(d.index[int(rs)].date()) if isinstance(d.index, pd.DatetimeIndex) else None
            except Exception:
                breakout_start = None
        elif run_start is not None:
            # HS/IHS: breakout must occur soon after the pattern completes (avoid months-late neckline breaks)
            if cand.pattern in ("HS_TOP", "IHS"):
                meta = cand.meta if isinstance(cand.meta, dict) else {}
                p_end = int(meta.get("pattern_end_i", -1)) if isinstance(meta, dict) else -1
                if p_end >= 0 and (int(run_start) - int(p_end)) > HS_MAX_BREAKOUT_LAG_BARS:
                    continue
            age = int((len(d) - 1) - int(run_start))
            if age > CONFIRMED_MAX_AGE_BARS:
                continue
            prefix = "CONFIRMED_"
            stage_status = "NEW" if age == 0 else "ONGOING"
            stage_age_bars = int(age)
            try:
                breakout_start = str(d.index[int(run_start)].date()) if isinstance(d.index, pd.DatetimeIndex) else None
            except Exception:
                breakout_start = None
        else:
            # Not confirmed today -> can only be EARLY (pre-break) or nothing.
            prefix, dist_atr = _classify_vs_level(close, level_now, atr_val, cand.direction, vol_ratio, clv)
            if prefix != "EARLY_":
                continue

                # Suppress EARLY resurfacing after a prior confirmed move that later reverted back through the trigger.
                # Prevents cases like PHM/PSA being flagged as EARLY long after an actual breakout happened.
                try:
                    meta = cand.meta if isinstance(cand.meta, dict) else {}
                    start_i = int(meta.get("pattern_end_i", meta.get("p3", meta.get("p2", 0))))
                except Exception:
                    start_i = 0
                if cand.pattern in ("HS_TOP", "IHS") and _historical_confirm_then_retest(cand, d, a, start_i):
                    if isinstance(debug, dict):
                        debug.setdefault("suppressed_early_retests", []).append({
                            "ticker": ticker,
                            "pattern": cand.pattern,
                            "dir": cand.direction,
                            "start_i": start_i,
                        })
                    continue
            # EARLY exits: staleness TTL (pattern-specific) + blow-off failure (2x ATR pierce without confirmation, then revert)
            if cand.pattern != "DEAD_CAT_BOUNCE":
                meta = cand.meta if isinstance(cand.meta, dict) else {}
                p_end = meta.get("pattern_end_i", meta.get("end_i", None))
                try:
                    if p_end is not None:
                        p_end_i = int(p_end)
                        age_from_end = int((len(d) - 1) - p_end_i)

                        # Staleness TTL:
                        # - HS/IHS: 20 trading sessions from RS (pattern_end_i is RS for HS/IHS)
                        # - RECT/TRIANGLE/BROADEN: 20 trading sessions since last validating touch (pattern_end_i)
                        # - others: fallback to generic cap
                        if cand.pattern in ("HS_TOP", "IHS"):
                            ttl = 20
                        elif ("RECT" in cand.pattern) or ("TRIANGLE" in cand.pattern) or ("BROADEN" in cand.pattern):
                            ttl = 20
                        else:
                            ttl = int(EARLY_MAX_AGE_FROM_PATTERN_END_BARS) if "EARLY_MAX_AGE_FROM_PATTERN_END_BARS" in globals() else int(EARLY_MAX_AGE_FROM_PATTERN_END_BARS)

                        if age_from_end > int(ttl):
                            continue

                        # Blow-off failure exit:
                        # If intraday pierces >= 2x ATR beyond the level without price-confirm close, then later reverts through the level,
                        # suppress resurfacing as EARLY.
                        if (cand.pattern in ("HS_TOP", "IHS")) or ("RECT" in cand.pattern) or ("TRIANGLE" in cand.pattern) or ("BROADEN" in cand.pattern):
                            if _historical_big_pierce_then_revert(cand, d, a, start_i=p_end_i, pierce_mult=2.0):
                                continue
                except Exception:
                    pass
# VP runway (distance to nearest opposing HVN) for CONFIRMED + VALIDATED

        # Exit rules (apply to all formations once CONFIRMED/VALIDATED):
        # - exit if price crosses back through the trigger/neckline
        # - or if the move gives back >= 2 ATR_ref from its peak excursion since run_start
        exit_run_start = None
        try:
            if prefix == "VALIDATED_" and 'vinfo' in locals() and vinfo is not None:
                exit_run_start = int(rs)
            elif prefix == "CONFIRMED_" and run_start is not None:
                exit_run_start = int(run_start)
        except Exception:
            exit_run_start = None
        if exit_run_start is not None:
            exi = _exit_check_giveback(cand, d, a, exit_run_start, len(d) - 1, giveback_atr=EXIT_GIVEBACK_ATR)
            if isinstance(exi, dict) and exi.get("exit"):
                if isinstance(debug, dict):
                    debug.setdefault("exit_events", []).append({
                        "ticker": ticker,
                        "signal": f"{prefix}{cand.pattern}_{cand.direction}",
                        "stage": "VALIDATED" if prefix == "VALIDATED_" else "CONFIRMED",
                        "reason": str(exi.get("reason")),
                        "peak_excess": exi.get("peak_excess"),
                        "cur_excess": exi.get("cur_excess"),
                        "atr_ref": exi.get("atr_ref"),
                    })
                continue
        if prefix in ("CONFIRMED_", "VALIDATED_"):
            try:
                vp_runway_pct, _z = _vp_runway_to_hvn_pct(d, close=close, direction=cand.direction, end_idx=len(d) - 1)
                if isinstance(_z, dict):
                    vp_zone_low = _safe_float(_z.get("low"))
                    vp_zone_high = _safe_float(_z.get("high"))
            except Exception:
                vp_runway_pct, vp_zone_low, vp_zone_high = None, None, None
# Dead-cat-bounce EARLY must be fresh (event-driven) or we suppress it
        if cand.pattern == "DEAD_CAT_BOUNCE" and prefix == "EARLY_":
            meta = cand.meta if isinstance(cand.meta, dict) else {}
            age_low = int(meta.get("age_from_event_low_bars", 999))
            age_bounce = int(meta.get("age_from_bounce_high_bars", 999))
            if age_low > DCB_EARLY_MAX_BARS or age_bounce > DCB_EARLY_MAX_FROM_BOUNCE:
                continue
        sigs.append(LevelSignal(
            ticker=ticker,
            signal=f"{prefix}{cand.pattern}_{cand.direction}",
            pattern=cand.pattern,
            direction=cand.direction,
            level=float(level_now),
            close=close,
            atr=atr_val,
            dist_atr=float(dist_atr),
            stage_status=stage_status,
            stage_age_bars=stage_age_bars,
            breakout_start=breakout_start,
            pct_today=pct_today,
            clv=clv,
            vol_ratio=vol_ratio,
            vp_hvn_runway_pct=vp_runway_pct,
            vp_hvn_zone_low=vp_zone_low,
            vp_hvn_zone_high=vp_zone_high,
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
    # Debug: stage counts
    if isinstance(debug, dict):
        for s in sigs:
            sid = str(getattr(s, "signal", ""))
            if sid.startswith("EARLY_"):
                debug["signals_early"] = int(debug.get("signals_early", 0)) + 1
            elif sid.startswith("CONFIRMED_"):
                debug["signals_conf"] = int(debug.get("signals_conf", 0)) + 1
            elif sid.startswith("VALIDATED_"):
                debug["signals_val"] = int(debug.get("signals_val", 0)) + 1
        debug["signals_total"] = int(debug.get("signals_total", 0)) + int(len(sigs))
    return sigs
def _hs_indices_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    """Extract (LS_i, H_i, RS_i) from HS/IHS meta points."""
    if not isinstance(meta, dict):
        return None
    pts = meta.get("points")
    if not isinstance(pts, list):
        return None
    idx = {}
    for p in pts:
        if isinstance(p, dict):
            lab = str(p.get("label") or p.get("name") or "").upper()
            if lab in ("LS", "H", "RS"):
                try:
                    idx[lab] = int(p.get("i"))
                except Exception:
                    pass
    if all(k in idx for k in ("LS", "H", "RS")):
        return (idx["LS"], idx["H"], idx["RS"])
    return None
def _hs_meta_passes_guardrails(d: pd.DataFrame, pattern: str, meta: Optional[Dict[str, Any]]) -> bool:
    """Return True if HS/IHS meta is geometrically valid on detector window d."""
    tri = _hs_indices_from_meta(meta)
    if tri is None:
        return False
    p1, p2, p3 = tri
    inverse = (str(pattern).upper() == "IHS")
    g = _hs_geometry_diagnostics(
        d, p1, p2, p3, inverse=inverse,
        local_window=HS_LOCAL_WINDOW,
        symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO,
        valley_atr_mult=HS_VALLEY_ATR_MULT,
        shoulder_valley_mult=HS_SHOULDER_VALLEY_ATR_MULT,
    )
    return bool(g.get("pass_all", False))

def _focus_validation_gate_history(
    ticker: str,
    df0: pd.DataFrame,
    state: Optional[Dict[str, Any]] = None,
    lookback_bars: int = 30,
) -> List[Dict[str, Any]]:
    """Walk recent bars for a ticker's best candidate, checking all 3 confirmation gates individually.

    Returns a list of per-bar dicts:
      date, close, level, dist_atr, price_ok, clv, clv_ok, vol_ratio, vol_ok,
      all3 (confirmed), pricevol (day1-eligible), lifecycle_label, exit_reason
    """
    if df0 is None or df0.empty or len(df0) < 80:
        return []
    d0 = df0.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d0) < 80:
        return []
    d = d0.tail(LOOKBACK_DAYS).copy()
    d = _latest_completed_close_df(d)
    if len(d) < 80:
        return []
    a = atr(d, ATR_N).astype(float)
    # Detect candidates (with state carry-forward)
    candidates = detect_pattern_candidates(d)
    if isinstance(state, dict):
        mem = state.get("hs_geom", {}).get(ticker)
        have_hs = any(getattr(c, "pattern", "") in ("HS_TOP", "IHS") for c in candidates)
        if (not have_hs) and isinstance(mem, dict):
            meta2 = _reindex_meta_to_df(mem.get("meta", {}), d)
            if meta2 is not None:
                candidates.append(PatternCandidate(
                    pattern=str(mem.get("pattern", "")),
                    direction=str(mem.get("direction", "")),
                    level=float(mem.get("level", 0.0)),
                    meta=meta2,
                ))
        # Band geometry carry-forward
        mem_b = state.get("band_geom", {}).get(ticker)
        have_band = any(isinstance(getattr(c, "meta", None), dict) and str(c.meta.get("annot_type", "")) == "band" for c in candidates)
        if (not have_band) and isinstance(mem_b, dict):
            try:
                meta2b = _reindex_meta_to_df(mem_b.get("meta", {}), d)
                if meta2b is not None:
                    pat_b = str(mem_b.get("pattern", ""))
                    candidates.append(PatternCandidate(pattern=pat_b, direction="BREAKOUT", level=0.0, meta=meta2b))
                    candidates.append(PatternCandidate(pattern=pat_b, direction="BREAKDOWN", level=0.0, meta=meta2b))
            except Exception:
                pass
    # Filter by guardrails
    try:
        candidates = [
            c for c in candidates
            if not (getattr(c, "pattern", "") in ("HS_TOP", "IHS") and
                    (not _hs_meta_passes_guardrails(d, getattr(c, "pattern", ""), getattr(c, "meta", None))))
        ]
    except Exception:
        pass
    if not candidates:
        return []
    # Pick best candidate (highest gate score on last bar, prefer HS/IHS)
    end = len(d) - 1
    close_end = float(d["Close"].iloc[end])
    atr_end = float(a.iloc[end]) if len(a) > end and np.isfinite(a.iloc[end]) else 1.0
    best_cand = None
    best_score = -1
    for cnd in candidates:
        try:
            lvl = float(_level_at_bar(cnd, d, end))
            dist = (close_end - lvl) / max(atr_end, 1e-9)
            p_ok = (dist >= ATR_CONFIRM_MULT) if cnd.direction == "BREAKOUT" else (dist <= -ATR_CONFIRM_MULT)
            clv_v = _clv_at_bar(d, end)
            c_ok = (clv_v >= CLV_BREAKOUT_MIN) if cnd.direction == "BREAKOUT" else (clv_v <= CLV_BREAKDOWN_MAX)
            v_v = float(d["Volume"].iloc[end]) if "Volume" in d.columns else 0.0
            avg20 = float(pd.to_numeric(d["Volume"].iloc[max(0, end-21):end], errors="coerce").tail(20).mean()) if "Volume" in d.columns else 1.0
            vr = v_v / max(avg20, 1e-9) if np.isfinite(avg20) and np.isfinite(v_v) else 0.0
            v_ok = vr >= VOL_CONFIRM_MULT
            sc = int(p_ok) + int(c_ok) + int(v_ok)
            # Boost HS/IHS to prefer them
            if cnd.pattern in ("HS_TOP", "IHS"):
                sc += 10
            if sc > best_score:
                best_score = sc
                best_cand = cnd
        except Exception:
            continue
    if best_cand is None:
        return []
    cand = best_cand
    # Walk the last `lookback_bars` bars
    start = max(0, end - lookback_bars + 1)
    rows: List[Dict[str, Any]] = []
    # Pre-compute: find the most recent confirmed run start for lifecycle labeling
    run_start = None
    try:
        run_start_v = _confirm_run_start(cand, d, a)
        if run_start_v is not None:
            run_start = int(run_start_v)
    except Exception:
        pass
    # Also check validated stage
    vinfo = None
    try:
        vinfo = _validated_stage(cand, d, a, end)
    except Exception:
        pass
    # Walk bars
    for i in range(start, end + 1):
        try:
            close_i = float(d["Close"].iloc[i])
            lvl_i = float(_level_at_bar(cand, d, i))
            atr_i = float(a.iloc[i]) if i < len(a) and np.isfinite(a.iloc[i]) else float("nan")
            if not np.isfinite(atr_i) or atr_i <= 0:
                atr_i = max(abs(lvl_i) * 0.01, 1e-6)
            dist_i = (close_i - lvl_i) / atr_i
            # Price gate
            if cand.direction == "BREAKOUT":
                price_ok = dist_i >= ATR_CONFIRM_MULT
            else:
                price_ok = dist_i <= -ATR_CONFIRM_MULT
            # CLV gate
            clv_i = _clv_at_bar(d, i)
            if cand.direction == "BREAKOUT":
                clv_ok = clv_i >= CLV_BREAKOUT_MIN
            else:
                clv_ok = clv_i <= CLV_BREAKDOWN_MAX
            # Volume gate
            vol_ratio_i = float("nan")
            vol_ok = False
            if "Volume" in d.columns:
                v_i = float(d["Volume"].iloc[i])
                if i >= 21:
                    avg20_i = float(pd.to_numeric(d["Volume"].iloc[i-21:i-1], errors="coerce").tail(20).mean())
                elif i > 1:
                    avg20_i = float(pd.to_numeric(d["Volume"].iloc[:i], errors="coerce").tail(20).mean())
                else:
                    avg20_i = float("nan")
                if np.isfinite(avg20_i) and avg20_i > 0 and np.isfinite(v_i):
                    vol_ratio_i = v_i / avg20_i
                    vol_ok = vol_ratio_i >= VOL_CONFIRM_MULT
            all3 = price_ok and clv_ok and vol_ok
            pricevol = price_ok and vol_ok
            # Date
            dt_str = str(pd.Timestamp(d.index[i]).date()) if isinstance(d.index, pd.DatetimeIndex) else str(i)
            row: Dict[str, Any] = {
                "bar": i,
                "date": dt_str,
                "close": close_i,
                "level": lvl_i,
                "dist_atr": dist_i,
                "price_ok": price_ok,
                "clv": clv_i,
                "clv_ok": clv_ok,
                "vol_ratio": vol_ratio_i,
                "vol_ok": vol_ok,
                "all3": all3,
                "pricevol": pricevol,
            }
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return []
    # Label lifecycle phases and find exit reason
    # Find the most recent confirmation run start by scanning the rows
    # A "confirmed run" starts at the first bar where all 3 gates hold,
    # preceded by a bar where they don't.
    run_start_idx = None
    for j, r in enumerate(rows):
        if r["all3"]:
            if j == 0 or not rows[j-1]["all3"]:
                run_start_idx = j  # new run starts here
    # Label each row
    for j, r in enumerate(rows):
        label = "—"
        if run_start_idx is not None:
            day = j - run_start_idx
            if day < 0:
                label = "PRE"
            elif day == 0:
                label = "DAY 0 (CONFIRMED)" if r["all3"] else "—"
            elif day == 1:
                if r["pricevol"]:
                    label = "DAY 1 (ONGOING)"
                else:
                    if not r["price_ok"]:
                        label = "DAY 1 — PRICE LOST"
                    elif not r["vol_ok"]:
                        label = "DAY 1 — VOLUME LOST"
                    else:
                        label = "DAY 1 — FAILED"
            elif day == 2:
                if r["all3"]:
                    label = "DAY 2 (VALIDATED)"
                else:
                    if not r["price_ok"]:
                        label = "DAY 2 — PRICE LOST"
                    elif not r["clv_ok"]:
                        label = "DAY 2 — CLV LOST"
                    elif not r["vol_ok"]:
                        label = "DAY 2 — VOLUME LOST"
                    else:
                        label = "DAY 2 — FAILED"
            else:
                # Day 3+
                if r["price_ok"]:
                    label = f"DAY {day} (VALIDATED ONGOING)"
                else:
                    label = f"DAY {day} — PRICE RECROSS"
        r["lifecycle"] = label
    # Determine current status / exit reason
    exit_summary = ""
    if run_start_idx is not None:
        last = rows[-1]
        last_day = len(rows) - 1 - run_start_idx
        if last_day >= 2 and rows[run_start_idx + 2]["all3"] if (run_start_idx + 2) < len(rows) else False:
            # Was validated
            if last["price_ok"]:
                exit_summary = "VALIDATED — still active"
            else:
                # Find the first bar after validation where price recrossed
                for k in range(run_start_idx + 2, len(rows)):
                    if not rows[k]["price_ok"]:
                        exit_summary = f"EXIT — Price recross on {rows[k]['date']} (day {k - run_start_idx})"
                        break
                if not exit_summary:
                    exit_summary = "VALIDATED — still active"
            # Also check giveback
            try:
                exi = _exit_check_giveback(cand, d, a, rows[run_start_idx]["bar"], end, giveback_atr=EXIT_GIVEBACK_ATR)
                if isinstance(exi, dict) and exi.get("exit"):
                    reason = str(exi.get("reason", ""))
                    pk = exi.get("peak_excess", float("nan"))
                    cur = exi.get("cur_excess", float("nan"))
                    exit_summary = f"EXIT — {reason} (peak={pk:.2f} ATR, now={cur:.2f} ATR)"
            except Exception:
                pass
        elif last_day == 1:
            r1 = rows[run_start_idx + 1] if (run_start_idx + 1) < len(rows) else None
            if r1 and r1["pricevol"]:
                exit_summary = "CONFIRMED ONGOING — awaiting day 2"
            elif r1:
                if not r1["price_ok"]:
                    exit_summary = "EXIT — Price not sustained on day 1"
                else:
                    exit_summary = "EXIT — Volume faded on day 1"
            else:
                exit_summary = "CONFIRMED — day 0 only"
        elif last_day == 0:
            exit_summary = "CONFIRMED — day 0 (today)"
        else:
            exit_summary = "PRE-CONFIRMATION"
    else:
        exit_summary = "No confirmed run found in window"
    # Attach summary
    for r in rows:
        r["_exit_summary"] = exit_summary
        r["_pattern"] = cand.pattern
        r["_direction"] = cand.direction
    return rows

def _debug_gates_for_ticker(ticker: str, df0: pd.DataFrame, state: Optional[Dict[str, Any]] = None, max_candidates: int = 6) -> Dict[str, Any]:
    """Diagnostics for a ticker: last-bar metrics and why it did/didn't confirm."""
    out: Dict[str, Any] = {"Ticker": ticker}
    if df0 is None or df0.empty:
        out["note"] = "no data"
        return out
    d0 = df0.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if d0.empty or len(d0) < 5:
        out["note"] = "insufficient bars"
        return out
    # Use trading-bar lookback everywhere so HS/IHS meta indices align deterministically.
    d = d0.tail(LOOKBACK_DAYS).copy()
    d = _latest_completed_close_df(d)
    if d.empty or len(d) < 5:
        out["note"] = "empty slice"
        return out
    end = len(d) - 1
    close = float(d["Close"].iloc[end])
    out["LastDate"] = str(pd.Timestamp(d.index[end]).date())
    out["Close"] = close
    try:
        out["Day%"] = (float(d["Close"].iloc[end]) / float(d["Close"].iloc[end-1]) - 1.0) * 100.0
    except Exception:
        out["Day%"] = float("nan")
    out["CLV"] = _clv_at_bar(d, end)
    try:
        v = float(d["Volume"].iloc[end]) if "Volume" in d.columns else float("nan")
        avg20 = float(pd.to_numeric(d["Volume"].iloc[max(0, end-21):end], errors="coerce").tail(20).mean()) if "Volume" in d.columns else float("nan")
        out["VolRatio"] = v / avg20 if avg20 and np.isfinite(avg20) and np.isfinite(v) else float("nan")
    except Exception:
        out["VolRatio"] = float("nan")
    a = atr(d, ATR_N).astype(float)
    atr_v = float(a.iloc[end]) if len(a) and np.isfinite(a.iloc[end]) else float("nan")
    out["ATR"] = atr_v
    candidates = detect_pattern_candidates(d)
    if isinstance(state, dict):
        mem = state.get("hs_geom", {}).get(ticker)
        have_hs = any(getattr(cnd, "pattern", "") in ("HS_TOP","IHS") for cnd in candidates)
        if (not have_hs) and isinstance(mem, dict):
            meta2 = _reindex_meta_to_df(mem.get("meta", {}), d)
            if meta2 is not None:
                candidates.append(PatternCandidate(
                    pattern=str(mem.get("pattern","")),
                    direction=str(mem.get("direction","")),
                    level=float(mem.get("level",0.0)),
                    meta=meta2
                ))
    # HARD FAIL: never keep HS/IHS candidates whose geometry does not pass deterministic guardrails
    # on the same detector window used for level evaluation.
    try:
        candidates = [
            c for c in candidates
            if not (getattr(c, "pattern", "") in ("HS_TOP", "IHS") and (not _hs_meta_passes_guardrails(d, getattr(c, "pattern", ""), getattr(c, "meta", None))))
        ]
    except Exception:
        pass
    out["Cand#"] = len(candidates)
    rows = []
    for cnd in candidates[:max_candidates]:
        try:
            lvl = float(_level_at_bar(cnd, d, end))
            dist = (close - lvl) / atr_v if atr_v and np.isfinite(atr_v) else float("nan")
            price_ok = (dist >= ATR_CONFIRM_MULT) if cnd.direction == "BREAKOUT" else (dist <= -ATR_CONFIRM_MULT)
            clv_ok = (out["CLV"] >= CLV_BREAKOUT_MIN) if cnd.direction == "BREAKOUT" else (out["CLV"] <= CLV_BREAKDOWN_MAX)
            vol_ok = (out["VolRatio"] >= VOL_CONFIRM_MULT) if np.isfinite(out["VolRatio"]) else False
            hs_lag = ""
            if cnd.pattern in ("HS_TOP","IHS"):
                try:
                    pe = int((cnd.meta or {}).get("pattern_end_i"))
                    hs_lag = str(int(end - pe))
                except Exception:
                    hs_lag = ""
            rows.append({
                "pattern": cnd.pattern,
                "dir": cnd.direction,
                "distATR": dist,
                "level": lvl,
                "price_ok": price_ok,
                "clv_ok": clv_ok,
                "vol_ok": vol_ok,
                "hs_lag": hs_lag,
                "meta": cnd.meta,
            })
        except Exception:
            continue
    best = None
    best_score = -1
    for r in rows:
        sc = int(r["price_ok"]) + int(r["clv_ok"]) + int(r["vol_ok"])
        if sc > best_score:
            best_score = sc
            best = r
    out["Best"] = best
    out["Top"] = rows
    return out
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
            label = str(ln.get("label") or "")
            # For necklines, extend the line from T1/R1 to the chart right edge
            # so the projection remains visible beyond T2/R2.
            if label.lower() == "neckline":
                try:
                    xlim_right = matplotlib.dates.num2date(ax.get_xlim()[1])
                    x_right = pd.Timestamp(xlim_right).tz_localize(None)
                    dt_total = (t2 - t1).total_seconds()
                    if abs(dt_total) > 0:
                        slope_per_sec = (y2 - y1) / dt_total
                        dt_ext = (x_right - t1).total_seconds()
                        y_right = y1 + slope_per_sec * dt_ext
                        # Draw solid segment T1→T2 and dashed extension T2→right
                        ax.plot([t1, t2], [y1, y2], linestyle="--", linewidth=1.2,
                                color="red", alpha=0.85)
                        ax.plot([t2, x_right], [y2, y_right], linestyle=":",
                                linewidth=1.0, color="red", alpha=0.5)
                        ax.text(x_right, y_right, f" {label}", va="bottom",
                                fontsize=8, color="red", alpha=0.7)
                    else:
                        ax.plot([t1, t2], [y1, y2], linestyle="--", linewidth=1)
                        ax.text(t2, y2, f" {label}", va="bottom")
                except Exception:
                    ax.plot([t1, t2], [y1, y2], linestyle="--", linewidth=1)
                    ax.text(t2, y2, f" {label}", va="bottom")
            else:
                ax.plot([t1, t2], [y1, y2], linestyle="--", linewidth=1)
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
def plot_signal_chart(ticker: str, df: pd.DataFrame, sig: LevelSignal, name_resolver=None) -> Optional[str]:
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
    # Display label for charts: Company (TICKER)
    try:
        nm = str(name_resolver(ticker) or "").strip() if callable(name_resolver) else ""
    except Exception:
        nm = ""
    label = f"{nm} ({ticker})" if nm and nm.upper() != str(ticker).upper() else str(ticker)
    def placeholder(reason: str) -> str:
        fig = plt.figure(figsize=(10.5, 5.0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.75, f"{label}", fontsize=16, weight="bold", transform=ax.transAxes)
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
        # Use the same detector-style window as pattern indices (tail LOOKBACK_DAYS).
        # This avoids index/date mismatches where annotations point to the wrong bar.
        d = d_full.tail(LOOKBACK_DAYS).copy()
        d = _latest_completed_close_df(d)
        if len(d) < CHART_MIN_BARS:
            d = d_full.tail(CHART_MIN_BARS).copy()
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
        # For HS/IHS, the trigger is the (possibly sloped) Neckline line from meta.
        # We compute 'trigger_now' from that line at the last bar and use it for confirmation threshold.
        trigger_now = sig.level
        try:
            if isinstance(getattr(sig, 'meta', None), dict):
                for _ln in (sig.meta.get('lines') or []):
                    if isinstance(_ln, dict) and str(_ln.get('label','')).lower() == 'neckline':
                        i1 = int(_ln.get('i1')); i2 = int(_ln.get('i2'))
                        y1 = float(_ln.get('y1')); y2 = float(_ln.get('y2'))
                        a_fit, b_fit = _line_fit(np.array([float(i1), float(i2)]), np.array([float(y1), float(y2)]))
                        trigger_now = float(_line_eval(a_fit, b_fit, float(len(d) - 1)))
                        break
        except Exception:
            pass
        # Confirm line per rule
        direction = 1 if 'BREAKOUT' in sig.signal else -1 if 'BREAKDOWN' in sig.signal else 0
        confirm = float(trigger_now + direction * 0.5 * atr_last)

        # Draw trigger line: HS/IHS uses neckline (drawn by meta); other patterns use a horizontal trigger line.
        if not (sig.pattern in ('HS_TOP', 'IHS') and isinstance(getattr(sig, 'meta', None), dict) and any(isinstance(_ln, dict) and str(_ln.get('label','')).lower()=='neckline' for _ln in (sig.meta.get('lines') or []))):
            ax.axhline(sig.level, linestyle='-.', linewidth=1)
            ax.text(d.index[-1], sig.level, ' Trigger', va='bottom')
        else:
            # HS/IHS: keep the sloped Neckline line on the chart AND draw today's trigger level as a horizontal line.
            ax.axhline(trigger_now, linestyle='-.', linewidth=1)
            ax.scatter([d.index[-1]], [trigger_now], s=30)
            ax.text(d.index[-1], trigger_now, ' Trigger', va='bottom')
        ax.axhline(confirm, linestyle=':', linewidth=1)
        ax.text(d.index[-1], confirm, ' Confirm (±0.5 ATR)', va='bottom')
        # Pattern markings
        close = d["Close"].astype(float).values
        high = d["High"].astype(float).values
        low = d["Low"].astype(float).values
        used_meta_annotation = _annotate_from_signal_meta(ax, sig)
        if not used_meta_annotation:
            # Minimal fallbacks: avoid drawing helper pivot labels (R1/R2/T1/T2) and extra lines.
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
        title = f"{label} | {sig.signal}"
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
def _pct_change_n(c: pd.Series, n: int) -> Optional[float]:
    c = pd.to_numeric(c, errors="coerce").dropna()
    if len(c) <= n:
        return None
    prev = float(c.iloc[-1 - n])
    last = float(c.iloc[-1])
    if prev == 0:
        return None
    return (last / prev - 1.0) * 100.0
def _pct_ytd(c: pd.Series) -> Optional[float]:
    c = pd.to_numeric(c, errors="coerce").dropna()
    if c.empty:
        return None
    try:
        year = datetime.now().year
        start = pd.Timestamp(year=year, month=1, day=1)
        c_y = c[c.index >= start]
        if c_y.empty:
            return None
        base = float(c_y.iloc[0])
        last = float(c.iloc[-1])
        if base == 0:
            return None
        return (last / base - 1.0) * 100.0
    except Exception:
        return None
def build_watchlist_performance_section_md(
    ohlcv: Dict[str, pd.DataFrame],
    sector_resolver,
    name_resolver=None,
    country_resolver=None,
) -> str:
    """Section 6: Watchlist performance (all tickers) — grouped by watchlist segments.
    Columns (as requested):
      Name of Company | Ticker | Country | Sector | Close | Day% | CLV | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M
    """
    md: List[str] = []
    md.append("## 6) Watchlist performance (all tickers)\n")
    md.append("Columns: **Name of Company | Ticker | Country | Sector | Close | Day% | CLV | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M**\n")
    def _safe_name(t: str) -> str:
        # Name overrides (full caps) + commodities display names
        try:
            if t in NAME_OVERRIDES:
                return str(NAME_OVERRIDES[t]).upper()
            base = _base_ticker(t)
            if base in NAME_OVERRIDES:
                return str(NAME_OVERRIDES[base]).upper()
            if t in COMMODITY_NAME_OVERRIDES:
                return str(COMMODITY_NAME_OVERRIDES[t]).upper()
            if base in COMMODITY_NAME_OVERRIDES:
                return str(COMMODITY_NAME_OVERRIDES[base]).upper()
        except Exception:
            pass
        if callable(name_resolver):
            try:
                return str(name_resolver(t) or "").upper()
            except Exception:
                return ""
        return 
    def _safe_country(t: str) -> str:
        if callable(country_resolver):
            try:
                return str(country_resolver(t) or "")
            except Exception:
                return ""
        return ""
    def _safe_sector(t: str) -> str:
        if callable(sector_resolver):
            try:
                return str(sector_resolver(t) or "")
            except Exception:
                return ""
        return ""
    def _clv_bar(df: pd.DataFrame) -> float:
        try:
            hi = float(df["High"].iloc[-1]); lo = float(df["Low"].iloc[-1]); cl = float(df["Close"].iloc[-1])
            if hi > lo:
                v = (2.0*cl - hi - lo) / (hi - lo)
                return float(max(-1.0, min(1.0, v)))
        except Exception:
            pass
        return float("nan")
    def _vol_ratio(df: pd.DataFrame) -> float:
        try:
            v = float(df["Volume"].iloc[-1])
            if len(df) >= 21:
                avg20_prior = float(df["Volume"].iloc[-21:-1].mean())
            else:
                avg20_prior = float(df["Volume"].tail(20).mean())
            if avg20_prior and np.isfinite(avg20_prior) and np.isfinite(v):
                return float(v / avg20_prior)
        except Exception:
            pass
        return float("nan")
    def _pct_n(series: pd.Series, n: int) -> float:
        try:
            s = series.dropna()
            if len(s) <= n:
                return float("nan")
            return float((float(s.iloc[-1]) / float(s.iloc[-(n+1)]) - 1.0) * 100.0)
        except Exception:
            return float("nan")
    def _atr_delta14(df: pd.DataFrame) -> float:
        try:
            a = atr(df, ATR_N).dropna()
            if len(a) < 15:
                return float("nan")
            a_now = float(a.iloc[-1])
            a_prev = float(a.iloc[-15])
            if a_prev and np.isfinite(a_prev) and np.isfinite(a_now):
                return float((a_now / a_prev - 1.0) * 100.0)
        except Exception:
            pass
        return float("nan")
    # Keep the original segment order from WATCHLIST_GROUPS
    for seg, tickers in WATCHLIST_GROUPS.items():
        rows: List[Dict[str, Any]] = []
        for t in tickers:
            df = ohlcv.get(t)
            if df is None or df.empty:
                continue
            d = df.dropna(subset=["Open","High","Low","Close"]).copy()
            if d.empty:
                continue
            close_s = d["Close"].astype(float)
            close_last = float(close_s.iloc[-1])
            rows.append({
                "Name of Company": _safe_name(t),
                "Ticker": display_ticker(t),
                "Country": _safe_country(t),
                "Sector": _safe_sector(t),
                "Close": close_last,
                "Day%": _pct_n(close_s, 1),
                "CLV": _clv_bar(d),
                "ATR(14)": float(atr(d, ATR_N).dropna().iloc[-1]) if not atr(d, ATR_N).dropna().empty else float("nan"),
                "ATR Δ14d": _atr_delta14(d),
                "Vol/AvgVol20": _vol_ratio(d),
                "1D": _pct_n(close_s, 1),
                "7D": _pct_n(close_s, 5),
                "1M": _pct_n(close_s, 21),
                "3M": _pct_n(close_s, 63),
            })
        md.append(f"### {seg}\n")
        if not rows:
            md.append("<em>None</em>\n")
            continue
        dfp = pd.DataFrame(rows)
        # Sort: strongest 1M then 3M within segment
        dfp["_1m"] = pd.to_numeric(dfp["1M"], errors="coerce")
        dfp["_3m"] = pd.to_numeric(dfp["3M"], errors="coerce")
        dfp = dfp.sort_values(by=["_1m","_3m","Ticker"], ascending=[False, False, True]).drop(columns=["_1m","_3m"])
        cols = ["Name of Company","Ticker","Country","Sector","Close","Day%","CLV","ATR(14)","ATR Δ14d","Vol/AvgVol20","1D","7D","1M","3M"]
        md.append(html_table_from_df(dfp, cols=cols, max_rows=200))
        md.append("")
    return "\n".join(md)
def signals_to_df(
    signals: List[LevelSignal],
    sector_resolver=None,
    name_resolver=None,
    country_resolver=None,
) -> pd.DataFrame:
    cols = ["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "HVN Runway%", "Chart"]
    if not signals:
        return pd.DataFrame(columns=cols)
    rows = []
    for s in signals:
        cat = ""
        try:
            if callable(sector_resolver):
                cat = str(sector_resolver(s.ticker) or "")
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
            "Ticker": display_ticker(s.ticker),
            "Country": country,
            "Sector": cat,
            "Signal": s.signal,
            "Close": s.close,
            "Day%": s.pct_today if s.pct_today is not None else np.nan,
            "Threshold": s.level,
            "CLV": s.clv if getattr(s, "clv", None) is not None else np.nan,
            "ATR(14)": s.atr,
            "Dist(ATR)": s.dist_atr,
            "Vol/AvgVol(20)": s.vol_ratio if getattr(s, "vol_ratio", None) is not None else np.nan,
            "HVN Runway%": s.vp_hvn_runway_pct if s.vp_hvn_runway_pct is not None else np.nan,
            "Chart": s.chart_path or "",
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
    if "HVN Runway%" in d.columns:
        d["HVN Runway%"] = pd.to_numeric(d["HVN Runway%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "Vol/AvgVol20" in d.columns:
        d["Vol/AvgVol20"] = pd.to_numeric(d["Vol/AvgVol20"], errors="coerce").map(lambda x: f"{x:.2f}×" if pd.notna(x) else "")
    if "CLV" in d.columns:
        d["CLV"] = pd.to_numeric(d["CLV"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    if "Day%" in d.columns:
        d["Day%"] = pd.to_numeric(d["Day%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    
    # Additional performance columns (watchlist section)
    for pc in ["Week%", "Month%", "3M%", "YTD%"]:
        if pc in d.columns:
            d[pc] = pd.to_numeric(d[pc], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "Last" in d.columns:
        d["Last"] = pd.to_numeric(d["Last"], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    if "Chart" in d.columns:
        d["Chart"] = d["Chart"].apply(lambda p: f"[chart]({p})" if isinstance(p, str) and p else "")
    out = d[cols]
    # Alignment: textual columns left, numeric-ish columns right
    left_cols = {"Name of Company", "Name", "Ticker", "Country", "Sector", "Signal", "Pattern", "Dir", "Chart", "Instrument", "Symbol", "symbol"}
    aligns = tuple("left" if c in left_cols else "right" for c in cols)
    return df_to_markdown_aligned(out, aligns=aligns, index=False)
def html_table_from_df(df: pd.DataFrame, cols: List[str], max_rows: int = 80) -> str:
    """HTML table for GitHub Pages (auto layout; horizontal scroll).
    Formats common numeric columns used across the report.
    """
    if df is None or df.empty:
        return "<em>None</em>"
    d = df.copy().head(max_rows)
    # Price-like columns
    for c in ["Close", "Level", "Threshold", "Last", "ATR(14)"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    # Distance columns
    if "Dist(ATR)" in d.columns:
        d["Dist(ATR)"] = pd.to_numeric(d["Dist(ATR)"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    # HVN runway
    if "HVN Runway%" in d.columns:
        d["HVN Runway%"] = pd.to_numeric(d["HVN Runway%"], errors="coerce").map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    # Volume ratio
    for vc in ["Vol/AvgVol20", "Vol/AvgVol(20)"]:
        if vc in d.columns:
            d[vc] = pd.to_numeric(d[vc], errors="coerce").map(lambda x: f"{x:.2f}×" if pd.notna(x) else "")
    # CLV
    if "CLV" in d.columns:
        d["CLV"] = pd.to_numeric(d["CLV"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    # Percent columns
    for pc in ["Day%", "1D", "7D", "1M", "3M", "Week%", "Month%", "YTD%"]:
        if pc in d.columns:
            d[pc] = pd.to_numeric(d[pc], errors="coerce").map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    # ATR delta percent
    if "ATR Δ14d" in d.columns:
        d["ATR Δ14d"] = pd.to_numeric(d["ATR Δ14d"], errors="coerce").map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    # Chart links
    if "Chart" in d.columns:
        def _mk(p):
            if isinstance(p, str) and p:
                u = _asset_url(p)
                return f'<a href="{u}" target="_blank">chart</a>'
            return ""
        d["Chart"] = d["Chart"].apply(_mk)
    cols_use = [c for c in cols if c in d.columns]
    if not cols_use:
        return "<em>None</em>"
    num_cols = {
        "Close","Last","Level","Threshold","Dist(ATR)","HVN Runway%","Vol/AvgVol20","Vol/AvgVol(20)","CLV",
        "Day%","1D","7D","1M","3M","Week%","Month%","YTD%","ATR(14)","ATR Δ14d"
    }
    thead = "<thead><tr>" + "".join([f"<th>{c}</th>" for c in cols_use]) + "</tr></thead>"
        # Color-code selected performance columns (requested for Section 6: 1D/7D/1M/3M).
    color_pct_cols = {"1D", "7D", "1M", "3M"}
    def _colorize_pct_cell(v: Any) -> Any:
        try:
            if not isinstance(v, str):
                return v
            s0 = v.strip()
            m = re.match(r"^([+-]?)(\d+(?:\.\d+)?)%$", s0)
            if not m:
                return v
            sign = m.group(1)
            if sign == "-":
                col = "#cf222e"
            elif sign == "+":
                col = "#1a7f37"
            else:
                col = ""
            if col:
                return f'<span style="color:{col};font-weight:600">{s0}</span>'
            return v
        except Exception:
            return v
    rows_html = []
    for _, r in d[cols_use].iterrows():
        tds = []
        for c in cols_use:
            v = r.get(c, "")
            if c in color_pct_cols:
                v = _colorize_pct_cell(v)
            cls = "num" if c in num_cols else "txt"
            if c in ("Name of Company", "Name"):
                cls = "wrap"
            tds.append(f'<td class="{cls}">{"" if v is None else v}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
    style = (
        "<style>"
        "table.tblauto{table-layout:auto;width:100%;border-collapse:collapse;margin:8px 0;}"
        "table.tblauto th,table.tblauto td{border:1px solid #e5e7eb;padding:6px 8px;vertical-align:top;}"
        "table.tblauto th{background:#f6f8fa;font-weight:600;white-space:nowrap;}"
        "table.tblauto td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}"
        "table.tblauto td.txt{white-space:nowrap;}"
        "table.tblauto td.wrap{white-space:normal;}"
        "</style>"
    )
    return style + f'<div style="overflow-x:auto"><table class="tblauto">{thead}{tbody}</table></div>'
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
def _dedupe_macro_cards(md_str: str) -> str:
    """Ensure VIX/EURUSD macro card images appear only once in the markdown.
    Some email/renderer paths can duplicate blocks; we keep the first occurrence."""
    if not md_str:
        return md_str
    # crude but effective: if more than one occurrence of each macro image src, remove later ones
    for key in ["macro_vix_5y.png", "macro_eurusd_5y.png"]:
        parts = md_str.split(key)
        if len(parts) <= 2:
            continue
        # rebuild keeping first occurrence only
        md_str = parts[0] + key + "".join(p.replace(key, "") for p in parts[1:])
    return md_str
def _strip_macro_images_for_email(md_str: str) -> str:
    """Remove macro card images from the email markdown to prevent iOS mail clients from duplicating them as attachments.
    Web report keeps the images in section 1; email version uses links only (no inline macro images).
    """
    if not md_str:
        return md_str
    out_lines = []
    for ln in md_str.splitlines():
        if "macro_vix_5y.png" in ln or "macro_eurusd_5y.png" in ln:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip() + "\n"

# ─────────────────────────────────────────────────────────────────────────────
# GC IGNITION SCORING PASS
# Called from main() after all_signals is built (Star 1 already fired).
# Scores Star 2 (dual EPS+Rev beat OR catalyst) and Star 3 (Star 2 + rev
# >=20% YoY + moat via OpenAI) for all Star 1 tickers. Writes results back
# to gc_state.json as ignition_signals[]. Section 7 reads from there.
# ─────────────────────────────────────────────────────────────────────────────
def _gc_openai_chat(prompt: str, max_tokens: int = 100):
    """Shared OpenAI chat call for GC scoring. Returns raw text or None."""
    import urllib.request as _req, json as _json, os as _os
    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    body = _json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": max_tokens,
    }).encode("utf-8")
    try:
        req = _req.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with _req.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [gc-openai] {e}")
        return None


def _gc_moat_assessment(ticker: str, name: str, sector: str, industry: str,
                         yoy: float, eps_streak: int, rev_streak: int) -> dict:
    """Ask OpenAI if this Star 3 candidate has a durable structural moat."""
    import json as _json
    prompt = (
        f"Company: {name} ({ticker})\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"Revenue YoY: {yoy:.1f}% | EPS beat streak: {eps_streak}Q | Rev growth streak: {rev_streak}Q\n\n"
        "Does this company have a DURABLE ECONOMIC MOAT — structural advantages making it "
        "significantly better than any direct competitor and hard to displace? "
        "Think network effects, switching costs, proprietary IP, cost scale, brand dominance. "
        "A fast-growing commodity producer or cyclical does NOT qualify.\n\n"
        "Respond ONLY as JSON: {\"moat\": true/false, \"rationale\": \"one sentence max 20 words\"}"
    )
    raw = _gc_openai_chat(prompt, max_tokens=80)
    if raw is None:
        return {"moat_confirmed": None, "moat_rationale": "API unavailable", "moat_source": "fallback"}
    try:
        parsed = _json.loads(raw)
        return {"moat_confirmed": bool(parsed.get("moat")),
                "moat_rationale": str(parsed.get("rationale", "")).strip(),
                "moat_source": "gpt-4o-mini"}
    except Exception:
        return {"moat_confirmed": None, "moat_rationale": f"parse error: {raw[:50]}", "moat_source": "fallback"}


def _gc_catalyst_assessment(headline: str, company: str) -> dict:
    """Ask OpenAI if a news headline is a massive company-thesis-changing catalyst."""
    import json as _json
    prompt = (
        f"Company: {company}\nHeadline: \"{headline}\"\n\n"
        "Is this a MASSIVE, company-thesis-changing catalyst? "
        "Qualifying: FDA approval, >$500M government contract, geopolitical demand shock, "
        "transformative M&A. Non-qualifying: analyst upgrades, minor product launches, "
        "earnings beats (scored separately).\n\n"
        "Respond ONLY as JSON: {\"massive\": true/false, \"rationale\": \"one sentence\"}"
    )
    raw = _gc_openai_chat(prompt, max_tokens=80)
    if raw is None:
        return {"confirmed": None, "rationale": "API unavailable"}
    try:
        parsed = _json.loads(raw)
        return {"confirmed": bool(parsed.get("massive")),
                "rationale": str(parsed.get("rationale", "")).strip()}
    except Exception:
        return {"confirmed": None, "rationale": f"parse error: {raw[:50]}"}


def _gc_company_brief(ticker: str, name: str, sector: str, moat_rationale: str) -> str:
    """Get a 2-sentence brief: what the company does + why the moat is real."""
    prompt = (
        f"Company: {name} ({ticker}), Sector: {sector}\n"
        f"Moat assessment: {moat_rationale}\n\n"
        "Write exactly 2 sentences: (1) what this company does, (2) why its competitive moat is real and durable. "
        "Be specific and concise. No preamble."
    )
    raw = _gc_openai_chat(prompt, max_tokens=120)
    return raw if raw else f"{name} operates in {sector}."


def run_gc_ignition_scoring(
    all_signals,
    ohlcv: dict,
    gc_state_path,
    name_resolver=None,
    sector_resolver=None,
) -> None:
    """
    GC ignition scoring pass — runs after Star 1 (technical signals) are found.

    For every ticker that fired a Star 1 signal, scores Star 2 and Star 3
    using gc_state.json earnings/revenue data. Writes results back to
    gc_state.json as ignition_signals[].

    Star 2: BOTH EPS beat (>=2Q) AND revenue growth (>=2Q) OR massive catalyst (OpenAI).
    Star 3: Star 2 + last-quarter revenue YoY >= 20% (quarterly source only) + moat (OpenAI).
    """
    import json as _json, datetime as _dt

    gc_state_path = Path(gc_state_path)
    if not gc_state_path.exists():
        print("[gc-scoring] gc_state.json not found — skipping ignition scoring")
        return

    try:
        gc_state = _json.loads(gc_state_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[gc-scoring] failed to load gc_state.json: {e}")
        return

    earnings_cache = gc_state.get("earnings_cache", {})
    if not earnings_cache:
        print("[gc-scoring] earnings_cache empty — skipping")
        return

    # Deduplicate tickers with Star 1 signals
    star1_tickers = list({s.ticker for s in all_signals})
    print(f"[gc-scoring] Star 1 tickers: {len(star1_tickers)}")

    EPS_MIN = 2  # minimum consecutive quarters for beat streak

    def _yoy_growth(data: dict):
        """Latest quarterly YoY revenue growth. Blocks FY/TTM/info sources."""
        rev = data.get("quarterly_revenue", [])
        if not rev:
            return None, "none"
        past = [r for r in rev if r.get("date") and r["date"] <= _dt.date.today().isoformat()]
        past_w_yoy = [r for r in past if r.get("revenue_yoy_growth") is not None
                      and r.get("revenue_source", "quarterly") not in ("info_fallback", "annual_estimated")]
        if not past_w_yoy:
            return None, "none"
        latest = sorted(past_w_yoy, key=lambda r: r["date"], reverse=True)[0]
        return latest.get("revenue_yoy_growth"), latest.get("revenue_source", "quarterly")

    def _eps_beat_streak(data: dict):
        """Returns (streak: int, used_proxy: bool).
        used_proxy=True means at least one quarter in the streak used YoY comparison
        instead of a real consensus EPS estimate."""
        dates = data.get("earnings_dates", [])
        past = sorted([d for d in dates if d.get("eps_reported") is not None
                       and (d.get("date") or "") <= _dt.date.today().isoformat()],
                      key=lambda d: d["date"], reverse=True)
        # Build prior-year EPS lookup for YoY proxy
        eps_by_qtr = {}
        for r in past:
            d_ = (r.get("date") or "")[:7]
            rep = r.get("eps_reported")
            if d_ and rep is not None:
                try: eps_by_qtr[d_] = float(rep)
                except Exception: pass

        streak = 0
        used_proxy = False
        for r in past:
            s = r.get("eps_surprise_pct")
            src = r.get("_eps_est_source", "")
            if s is None:
                try:
                    e = float(r.get("eps_estimate") or 0)
                    rep = float(r.get("eps_reported") or 0)
                    if abs(e) > 0.001:
                        s = (rep / e - 1.0) * 100
                except Exception: pass
            # YoY proxy: same quarter prior year
            if s is None:
                try:
                    d_ = (r.get("date") or "")[:7]
                    y, m = int(d_[:4]), int(d_[5:7])
                    rep = float(r.get("eps_reported") or 0)
                    for delta in [0, -1, 1, -2, 2]:
                        nm = m + delta; ny = y - 1 + (nm-1)//12; nm = ((nm-1)%12)+1
                        key = f"{ny:04d}-{nm:02d}"
                        if key in eps_by_qtr and eps_by_qtr[key] > 0.001:
                            s = (rep / eps_by_qtr[key] - 1.0) * 100
                            used_proxy = True
                            break
                except Exception: pass
            # Also flag if estimate came from proxy source
            if src in ("yoy_proxy",) or r.get("_method") == "income_stmt_derived":
                used_proxy = True
            if s is not None and float(s) > 0: streak += 1
            else: break
        return streak, used_proxy

    def _rev_beat_streak_for_data(data: dict):
        """Returns (streak: int, used_proxy: bool).
        used_proxy=True means at least one quarter used YoY fallback instead of
        consensus revenue estimate."""
        dates = data.get("earnings_dates", [])
        past = sorted([d for d in dates if d.get("eps_reported") is not None
                       and (d.get("date") or "") <= _dt.date.today().isoformat()],
                      key=lambda d: d["date"], reverse=True)
        # Build YYYY-MM -> yoy from quarterly_revenue
        rev_yoy = {}
        for qr in data.get("quarterly_revenue", []):
            dt_ = (qr.get("date") or "")[:7]
            g = qr.get("revenue_yoy_growth")
            if dt_ and g is not None:
                try: rev_yoy[dt_] = float(g)
                except Exception: pass

        def _rev_beat(r):
            """Returns (beat: bool, is_proxy: bool)"""
            try:
                r_est = r.get("revenue_estimate"); r_rep = r.get("revenue_reported")
                if r_est is not None and r_rep is not None and float(r_est) > 0:
                    return float(r_rep) > float(r_est), False  # real consensus
            except Exception: pass
            # Fallback: YoY proxy from income statement
            d_ = (r.get("date") or "")[:7]
            if d_ in rev_yoy: return rev_yoy[d_] > 0, True
            try:
                y, m = int(d_[:4]), int(d_[5:7])
                for delta in [-1, -2, 1, 2]:
                    nm = m + delta; ny = y + (nm - 1) // 12; nm = ((nm - 1) % 12) + 1
                    key = f"{ny:04d}-{nm:02d}"
                    if key in rev_yoy: return rev_yoy[key] > 0, True
            except Exception: pass
            return False, True  # can't determine — conservative False, but mark as proxy

        streak = 0
        used_proxy = False
        for r in past:
            rev_beat, rev_proxy = _rev_beat(r)
            if rev_proxy:
                used_proxy = True
            eps_streak_1, eps_proxy_1 = _eps_beat_streak({
                "earnings_dates": [r],
                "quarterly_revenue": data.get("quarterly_revenue", [])
            })
            eps_beat = eps_streak_1 > 0
            if rev_beat and eps_beat: streak += 1
            else: break
        return streak, used_proxy

    ignition_signals = []
    star1_count = len(star1_tickers)
    star2_count = 0
    star3_count = 0

    for ticker in star1_tickers:
        data = earnings_cache.get(ticker)
        if not data:
            continue

        eps_streak, eps_used_proxy = _eps_beat_streak(data)
        rev_streak, rev_used_proxy = _rev_beat_streak_for_data(data)
        yoy, rev_src = _yoy_growth(data)

        # ── Star 2 ────────────────────────────────────────────────────────────
        dual_beat = (eps_streak >= EPS_MIN and rev_streak >= EPS_MIN)
        massive_catalyst = False
        catalyst_rationale = ""

        if not dual_beat:
            # Check for massive catalyst
            for event in data.get("catalyst_events", []):
                if event.get("catalyst_tier") == 1:
                    company_name = (name_resolver(ticker) if name_resolver else ticker)
                    result = _gc_catalyst_assessment(event.get("headline", ""), company_name)
                    if result.get("confirmed"):
                        massive_catalyst = True
                        catalyst_rationale = result.get("rationale", "")
                        break

        # Single-beat data-gap fallback: if we have NO revenue consensus data
        # from any source (yfinance, investing.com, FMP, Finnhub) but EPS beat
        # is confirmed >=2Q, allow proceeding. Star 3 moat+20% YoY is the real gate.
        has_any_rev_consensus = any(
            r.get("revenue_estimate") is not None
            for r in data.get("earnings_dates", [])
            if r.get("eps_reported") is not None
        )
        data_gap_single = (
            not dual_beat and not massive_catalyst
            and not has_any_rev_consensus
            and eps_streak >= EPS_MIN
        )
        # Also allow revenue-only if strong (>= 2Q YoY growth) but EPS data weak
        rev_only_fallback = (
            not dual_beat and not massive_catalyst and not data_gap_single
            and rev_streak >= EPS_MIN and eps_streak < EPS_MIN
        )

        is_star2 = dual_beat or massive_catalyst or data_gap_single or rev_only_fallback
        if not is_star2:
            continue

        star2_count += 1
        sector = sector_resolver(ticker) if sector_resolver else ""
        name = name_resolver(ticker) if name_resolver else ticker

        if dual_beat: star2_via = "dual_beat"
        elif massive_catalyst: star2_via = "catalyst"
        elif data_gap_single: star2_via = "data_gap_eps_only"
        else: star2_via = "data_gap_rev_only"

        signal_entry = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "stars": 2,
            "eps_beat_streak": eps_streak,
            "revenue_beat_streak": rev_streak,
            "eps_beat_proxy": eps_used_proxy,   # True = YoY proxy used, no real consensus EPS estimate
            "rev_beat_proxy": rev_used_proxy,   # True = YoY proxy used, no real consensus rev estimate
            "yoy_growth": yoy,
            "revenue_source": rev_src,
            "star2_via": star2_via,
            "catalyst_rationale": catalyst_rationale,
            "moat_confirmed": None,
            "moat_rationale": "",
            "moat_source": "",
            "company_brief": "",
        }

        # ── Star 3 ────────────────────────────────────────────────────────────
        if yoy is not None and yoy >= 20.0 and rev_src not in ("info_fallback", "annual_estimated", "none"):
            info = data.get("info", {})
            industry = str(info.get("industry", ""))
            moat = _gc_moat_assessment(ticker, name, sector, industry, yoy, eps_streak, rev_streak)
            signal_entry["moat_confirmed"] = moat["moat_confirmed"]
            signal_entry["moat_rationale"] = moat["moat_rationale"]
            signal_entry["moat_source"] = moat["moat_source"]

            if moat.get("moat_confirmed"):
                signal_entry["stars"] = 3
                signal_entry["company_brief"] = _gc_company_brief(
                    ticker, name, sector, moat["moat_rationale"])
                star3_count += 1

        ignition_signals.append(signal_entry)

    print(f"[gc-scoring] Star 1: {star1_count} | Star 2: {star2_count} | Star 3: {star3_count}")

    # Write ignition_signals back to gc_state.json
    gc_state["ignition_signals"] = ignition_signals
    gc_state["last_ignition_score"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        gc_state_path.write_text(_json.dumps(gc_state, indent=2, default=str), encoding="utf-8")
        print(f"[gc-scoring] wrote {len(ignition_signals)} ignition signals to gc_state.json")
    except Exception as e:
        print(f"[gc-scoring] failed to write gc_state.json: {e}")


def build_gc_three_star_section_md(gc_state_path) -> str:
    """
    Section 7: Growth Compounders - Three-Star Signals.
    Reads ignition_signals from gc_state.json and renders a table + per-ticker
    brief for every ticker with stars == 3 (moat confirmed).

    Columns: Ticker | Name | Sector | Rev YoY | EPS beats | Rev beats | Star2 via | Moat rationale
    Followed by a one-paragraph brief per ticker generated by OpenAI.
    """
    import json as _json, os as _os, urllib.request as _req

    try:
        gc_state = _json.loads(Path(gc_state_path).read_text(encoding="utf-8"))
    except Exception:
        return ""

    signals = gc_state.get("ignition_signals", [])
    three_star = [s for s in signals if s.get("stars", 0) >= 3 and s.get("moat_confirmed")]
    if not three_star:
        return ""

    ec = gc_state.get("earnings_cache", {})

    # Sort by rev YoY descending
    three_star.sort(key=lambda s: s.get("yoy_growth") or 0, reverse=True)

    lines = []
    lines.append("## 7) Growth Compounders - Three-Star Signals\n")
    lines.append(
        "_Tickers that passed: (1) technical ignition, "
        "(2) dual EPS+Revenue beat >=2Q in a row or a massive catalyst, "
        "(3) revenue growth >=20% YoY with a durable moat confirmed by AI._"
    )
    lines.append("")

    # Summary table
    lines.append("| Ticker | Name | Sector | Rev YoY | EPS Q | Rev Q | Via | Moat |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |")
    # Note: ~ after Q count = YoY proxy used (no consensus estimate); ⚠proxy = both proxied
    for s in three_star:
        t = s.get("ticker", "")
        edata = ec.get(t, {})
        info = edata.get("info") or {}
        name   = info.get("short_name", t)
        sector = info.get("sector", "-")
        yoy    = s.get("yoy_growth")
        yoy_s  = f"{yoy:+.1f}%" if yoy is not None else "-"
        eps_q      = s.get("eps_beat_streak", 0)
        rev_q      = s.get("revenue_beat_streak", 0)
        via        = s.get("star2_via", "-")
        eps_proxy  = s.get("eps_beat_proxy", False)
        rev_proxy  = s.get("rev_beat_proxy", False)
        proxy_flag = ""
        if eps_proxy and rev_proxy: proxy_flag = " ⚠proxy"
        elif eps_proxy:             proxy_flag = " ⚠eps~"
        elif rev_proxy:             proxy_flag = " ⚠rev~"
        moat   = (s.get("moat_rationale") or "")[:60]
        lines.append(f"| **{t}** | {name} | {sector} | {yoy_s} | {eps_q}Q{' ~' if eps_proxy else ''} | {rev_q}Q{' ~' if rev_proxy else ''} | {via}{proxy_flag} | {moat} |")
    lines.append("")

    # Per-ticker brief via OpenAI
    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    for s in three_star:
        t = s.get("ticker", "")
        edata = ec.get(t, {})
        info  = edata.get("info") or {}
        name   = info.get("short_name", t)
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        yoy    = s.get("yoy_growth") or 0
        eps_q  = s.get("eps_beat_streak", 0)
        rev_q  = s.get("revenue_beat_streak", 0)
        moat   = s.get("moat_rationale", "")
        via    = s.get("star2_via", "")
        cat_r  = s.get("catalyst_rationale", "")

        lines.append(f"### {name} ({t})\n")
        lines.append(f"**Sector:** {sector} | **Industry:** {industry}\n")
        eps_proxy  = s.get("eps_beat_proxy", False)
        rev_proxy  = s.get("rev_beat_proxy", False)
        proxy_notes = []
        if eps_proxy: proxy_notes.append("EPS beats based on YoY comparison (no consensus EPS estimate available)")
        if rev_proxy: proxy_notes.append("Revenue beats based on YoY growth (no consensus revenue estimate available)")
        proxy_caveat = ""
        if proxy_notes:
            proxy_caveat = "  \n> ⚠ **Data caveat:** " + " | ".join(proxy_notes)

        lines.append(
            f"**Performance:** Rev YoY {yoy:+.1f}% | "
            f"EPS beat {eps_q}Q{'~' if eps_proxy else ''} | "
            f"Rev beat {rev_q}Q{'~' if rev_proxy else ''} | "
            f"Star 2 via: {via}"
            + (f" — {cat_r}" if cat_r else "")
            + proxy_caveat + "\n"
        )
        lines.append(f"**Moat:** {moat}\n")

        # OpenAI brief: what the company does + why the moat is real
        brief = None
        if api_key:
            prompt = (
                f"Company: {name} ({t}), {sector}/{industry}.\n"
                f"Revenue YoY growth: {yoy:.1f}%. EPS beat {eps_q} consecutive quarters.\n"
                f"Moat assessment: {moat}\n\n"
                "Write 2 sentences for a daily investment report:\n"
                "Sentence 1: what the company does and why it is growing so fast.\n"
                "Sentence 2: explain the moat in plain language - why no competitor can easily replicate this.\n"
                "Be specific, crisp, and avoid generic phrases like 'market leader' or 'strong brand'."
            )
            body = _json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 120,
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
                with _req.urlopen(req, timeout=20) as resp:
                    data = _json.loads(resp.read())
                brief = data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                brief = f"_(AI brief unavailable: {e})_"

        if brief:
            lines.append(f"{brief}\n")
        lines.append("")

    return "\n".join(lines)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "custom"], default=os.environ.get("MODE", "custom"))
    ap.add_argument("--max-tickers", type=int, default=int(os.environ.get("MAX_TICKERS", "0")))
    args = ap.parse_args()
    custom = get_custom_tickers()
    # Extend universe to commodities (no country; sector=Commodities)
    custom = sorted(set(custom + COMMODITY_TICKERS))
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

    # Load full MSCI universe (all 9 country CSVs, deduplicated, ghost-filtered)
    # via universe.py — single source of truth shared with gc_engine.
    msci_df = load_universe()   # already filters DEAD_MARKET_SUFFIXES + KNOWN_DEAD (v1.2.0)
    msci_all = set(msci_df["Ticker"].astype(str).tolist())

    # ── Option A: filter MSCI universe against gc_state.json active set ────────
    # gc_engine runs first and establishes which tickers are active (not inactive,
    # not below_min_mcap). scan.py uses that vetted set as its OHLCV download list.
    # This means: gc_engine MUST run before scan.py on each daily cycle.
    # New MSCI constituents not yet in gc_state pass through unfiltered — they get
    # one free fetch pass, then gc_engine trims them on the next run if below mcap.
    _gc_state_path = DOCS_DIR / "gc_state.json"
    _gc_active: set = set()
    _gc_inactive: set = set()
    if _gc_state_path.exists():
        try:
            import json as _gcj
            _gc_cache = _gcj.loads(_gc_state_path.read_text(encoding="utf-8")).get("earnings_cache", {})
            for _t, _v in _gc_cache.items():
                # Only exclude from OHLCV when gc_engine has CONFIRMED below floor.
                # Tickers with unknown mcap (_mcap_usd absent) are NOT excluded —
                # we only filter when we know and the value is confirmed below $2B USD.
                if _v.get("inactive") or _v.get("below_min_mcap"):
                    _gc_inactive.add(_t)
                else:
                    _gc_active.add(_t)
            print(f"[msci] gc_state loaded: {len(_gc_active)} active, {len(_gc_inactive)} inactive/below_mcap")
        except Exception as _e:
            print(f"[msci] gc_state load failed ({_e}) — using full MSCI universe as fallback")

    if _gc_active:
        # Use gc_state active set intersected with MSCI universe.
        # Always include base_universe tickers (watchlist/SPX/NDX) regardless of gc_state.
        # New MSCI tickers not yet seen by gc_engine are included for their first pass.
        _new_msci = msci_all - _gc_active - _gc_inactive   # not yet in gc_state at all
        msci_tickers = sorted(
            (msci_all & _gc_active) | _new_msci - set(base_universe)
        )
        _filtered = len(msci_all - _gc_active - _new_msci - set(base_universe))
        if _filtered:
            print(f"[msci] filtered {_filtered} inactive/below_mcap tickers from OHLCV download")
    else:
        # gc_state absent or empty (first run) — use full MSCI universe
        msci_tickers = sorted(msci_all - set(base_universe))

    if msci_tickers:
        print(f"[msci] {len(msci_tickers)} MSCI tickers for OHLCV download")
    else:
        print(f"[msci] no MSCI tickers loaded (base universe only)")

    tech_scan_universe = sorted(set(base_universe + msci_tickers))
    sector_resolver = build_sector_resolver(msci_df)
    company_name_for_ticker, country_for_ticker = build_company_country_resolvers(msci_df)
    # v83: ensure watchlist tickers show company names (not just ticker labels) in Section 4/6.
    # If a watchlist ticker is not in the MSCI mapping and has no NAME_OVERRIDES entry,
    # we fetch a lightweight name from yfinance (watchlist only) and use it as an override.
    watchlist_company_extra: Dict[str, str] = {}
    try:
        missing: List[str] = []
        for _t in sorted(watchlist_set):
            _t = str(_t or "").strip()
            if not _t:
                continue
            if _t in COMMODITY_NAME_OVERRIDES or _base_ticker(_t) in COMMODITY_NAME_OVERRIDES:
                continue
            nm0 = str(company_name_for_ticker(_t) or "").strip()
            # If the resolver collapses to ticker-like label, try to enrich.
            if (not nm0) or (nm0.upper() == _display_name(_t).upper()):
                missing.append(_t)
        # Safety: avoid hammering Yahoo; watchlist is small, but still guard.
        missing = missing[:80]
        if missing:
            for _t in missing:
                try:
                    info = yf.Ticker(_t).info or {}
                    nm = (info.get("shortName") or info.get("longName") or info.get("displayName") or "").strip()
                    if nm:
                        nmU = nm.upper()
                        watchlist_company_extra[_t] = nmU
                        watchlist_company_extra.setdefault(_base_ticker(_t), nmU)
                except Exception:
                    continue
    except Exception:
        watchlist_company_extra = {}
    if watchlist_company_extra:
        _base_name_resolver = company_name_for_ticker
        def company_name_for_ticker(ticker: str) -> str:
            t = str(ticker or "").strip()
            if not t:
                return ""
            base = _base_ticker(t)
            if t in watchlist_company_extra:
                return watchlist_company_extra[t]
            if base in watchlist_company_extra:
                return watchlist_company_extra[base]
            return _base_name_resolver(t)

    now = dt.datetime.now(dt.timezone.utc)
    global PUBLIC_BASE_URL, CACHE_BUST
    CACHE_BUST = now.strftime("%Y%m%d%H%M%S")
    PUBLIC_BASE_URL = _derive_public_base_url()
    # Backwards-compatible alias used throughout report generation
    def _cb_img(src: str) -> str:
        return _asset_url(src)
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
    # --- Download diagnostic: which tickers failed and is there a market/exchange pattern? ---
    _dl_ok = [t for t in tech_scan_universe if ohlcv.get(t) is not None and not ohlcv[t].empty]
    _dl_short = [t for t in tech_scan_universe if ohlcv.get(t) is not None and not ohlcv[t].empty and len(ohlcv[t]) < 80]
    _dl_fail = [t for t in tech_scan_universe if t not in ohlcv or ohlcv.get(t) is None or ohlcv[t].empty]
    _dl_usable = [t for t in _dl_ok if len(ohlcv[t]) >= 80]
    print(f"[download] universe={len(tech_scan_universe)} | downloaded={len(_dl_ok)} | usable(≥80 bars)={len(_dl_usable)} | short(<80 bars)={len(_dl_short)} | no_data={len(_dl_fail)}")
    # Break down failures by exchange suffix (proxy for market)
    def _exchange_suffix(t: str) -> str:
        if "." in t:
            parts = t.rsplit(".", 1)
            return f".{parts[-1]}" if len(parts) == 2 else "US"
        return "US"
    _fail_by_exch: Dict[str, int] = {}
    for t in _dl_fail:
        ex = _exchange_suffix(t)
        _fail_by_exch[ex] = _fail_by_exch.get(ex, 0) + 1
    _short_by_exch: Dict[str, int] = {}
    for t in _dl_short:
        ex = _exchange_suffix(t)
        _short_by_exch[ex] = _short_by_exch.get(ex, 0) + 1
    _total_by_exch: Dict[str, int] = {}
    for t in tech_scan_universe:
        ex = _exchange_suffix(t)
        _total_by_exch[ex] = _total_by_exch.get(ex, 0) + 1
    if _dl_fail:
        _fail_sorted = sorted(_fail_by_exch.items(), key=lambda x: x[1], reverse=True)
        print(f"[download] no_data by exchange: {', '.join(f'{k}:{v}/{_total_by_exch.get(k,0)}' for k, v in _fail_sorted[:15])}")
    if _dl_short:
        _short_sorted = sorted(_short_by_exch.items(), key=lambda x: x[1], reverse=True)
        print(f"[download] short(<80) by exchange: {', '.join(f'{k}:{v}/{_total_by_exch.get(k,0)}' for k, v in _short_sorted[:15])}")
    # Also resolve countries for failed tickers (via MSCI classification)
    _ctry_sorted: List[Tuple[str, int]] = []
    if _dl_fail and msci_df is not None and not msci_df.empty:
        _msci_country = dict(zip(msci_df["Ticker"].astype(str), msci_df["Country"].astype(str)))
        _fail_by_country: Dict[str, int] = {}
        for t in _dl_fail:
            ctry = _msci_country.get(t, "Unknown")
            _fail_by_country[ctry] = _fail_by_country.get(ctry, 0) + 1
        _ctry_sorted = sorted(_fail_by_country.items(), key=lambda x: x[1], reverse=True)
        print(f"[download] no_data by country: {', '.join(f'{k}:{v}' for k, v in _ctry_sorted[:15])}")
    # Sample failed tickers for inspection
    if _dl_fail:
        print(f"[download] sample failed tickers (first 20): {_dl_fail[:20]}")
    _dl_diag = {
        'universe': len(tech_scan_universe),
        'downloaded': len(_dl_ok),
        'usable': len(_dl_usable),
        'short': len(_dl_short),
        'no_data': len(_dl_fail),
        'fail_by_exch': sorted(_fail_by_exch.items(), key=lambda x: x[1], reverse=True)[:10] if _dl_fail else [],
        'fail_by_country': _ctry_sorted[:10],
    }
    # Load state early (used for HS/Band geometry carry-forward) + initialize debug counters
    state = load_state()
    debug: Dict[str, Any] = {
        'tickers_scanned': 0,
        'tickers_usable': 0,
        'cand_total': 0,
        'cand_by_pattern': {},
        'signals_early': 0,
        'signals_conf': 0,
        'signals_val': 0,
        'signals_total': 0,
        'hs_restored': 0,
        'band_restored': 0,
    }
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
    # After-hours movers (watchlist) via Yahoo quote endpoint (postMarketChangePercent)
    ah_all = fetch_watchlist_afterhours_movers_yahoo(mover_universe)
    print(f"[after-hours] yahoo_quote returned {len(ah_all)} after-hours movers")
    if not ah_all.empty:
        top = ah_all.sort_values("pct", ascending=False, key=lambda s: s.abs()).head(5)
        print(f"[after-hours] top movers: {list(zip(top['symbol'], top['pct'].round(2)))}")
    # Fallback: if Yahoo returns no extended-hours data for some tickers, supplement from StockAnalysis after-hours tables.
    try:
        fb_gain, fb_lose = fetch_afterhours_movers()
        fb = pd.concat([fb_gain, fb_lose], ignore_index=True) if (fb_gain is not None and fb_lose is not None) else pd.DataFrame()
        if fb is not None and not fb.empty:
            # fb schema: ['_symbol','_pct'] (pct already signed in losers table on StockAnalysis)
            fb2 = fb.copy()
            fb2["symbol"] = fb2["_symbol"].astype(str).str.strip()
            fb2["pct"] = pd.to_numeric(fb2["_pct"], errors="coerce")
            fb2 = fb2.dropna(subset=["pct"])
            # Keep only our mover universe symbols
            fb2 = fb2[fb2["symbol"].isin(set(mover_universe))][["symbol","pct"]]
            if ah_all is None or ah_all.empty:
                ah_all = fb2
            else:
                have = set(ah_all["symbol"].astype(str))
                fb2 = fb2[~fb2["symbol"].isin(have)]
                if not fb2.empty:
                    ah_all = pd.concat([ah_all, fb2], ignore_index=True)
    except Exception:
        pass
    ah_all = filter_movers(ah_all)
    ah_gf = ah_all[ah_all['pct'] >= MOVER_THRESHOLD_PCT].sort_values('pct', ascending=False)
    ah_lf = ah_all[ah_all['pct'] <= -MOVER_THRESHOLD_PCT].sort_values('pct', ascending=True)
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
# 4) # 4) Technical triggers
    all_signals: List[LevelSignal] = []
    # Watchlist/base universe: keep EARLY + CONFIRMED + VALIDATED (drives 4A + watchlist trend table)
    base_set = set(base_universe)
    for t in base_universe:
        debug['tickers_scanned'] += 1
        df = ohlcv.get(t)
        if df is None or df.empty:
            continue
        if len(df) >= 80:
            debug['tickers_usable'] += 1
        all_signals.extend(compute_signals_for_ticker(t, df, state=state, debug=debug))
    # MSCI expansion: include EARLY as well (to tune rules on a larger universe)
    if msci_tickers:
        for t in msci_tickers:
            debug['tickers_scanned'] += 1
            df = ohlcv.get(t)
            if df is None or df.empty:
                continue
            if len(df) >= 80:
                debug['tickers_usable'] += 1
            sigs = compute_signals_for_ticker(t, df, state=state, debug=debug)
            if sigs:
                all_signals.extend(sigs)
    validated = [s for s in all_signals if s.signal.startswith("VALIDATED_")]
    confirmed = [s for s in all_signals if s.signal.startswith("CONFIRMED_")]
    early = [s for s in all_signals if s.signal.startswith("EARLY_")]
    def rank_signal(s: LevelSignal) -> Tuple[int, float]:
        # Higher priority: VALIDATED > CONFIRMED > EARLY; tie-break by proximity to trigger
        if s.signal.startswith("VALIDATED_"):
            tier = 0
        elif s.signal.startswith("CONFIRMED_"):
            tier = 1
        else:
            tier = 2
        return (tier, abs(s.dist_atr))
    validated_sorted = sorted(validated, key=rank_signal)
    confirmed_sorted = sorted(confirmed, key=rank_signal)
    early_sorted = sorted(early, key=rank_signal)
    # Charts: VALIDATED (cap) then CONFIRMED (cap)
    val_charts = 0
    for s in validated_sorted:
        if val_charts >= int(MAX_CHARTS_VALIDATED):
            continue
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s, name_resolver=company_name_for_ticker)
        val_charts += 1
    conf_charts = 0
    for s in confirmed_sorted:
        if conf_charts >= int(MAX_CHARTS_CONFIRMED):
            continue
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s, name_resolver=company_name_for_ticker)
        conf_charts += 1
    # Charts: EARLY across all tickers (cap to keep report readable)
    early_charts = 0
    for s in early_sorted:
        if early_charts >= int(MAX_CHARTS_EARLY):
            continue
        s.chart_path = plot_signal_chart(s.ticker, ohlcv.get(s.ticker), s, name_resolver=company_name_for_ticker)
        early_charts += 1
# State diff (used only for EARLY "NEW" labeling + a changelog of signal IDs)
    # (state/debug already initialized above; do not reload here)
    prev_all = {"signals": state.get("signals", [])}
    prev_early = {"signals": state.get("early", [])}
    cur_all_ids = [f"{s.ticker}|{s.signal}" for s in all_signals]
    cur_early_ids = [f"{s.ticker}|{s.signal}" for s in early_sorted]
    state["signals"] = cur_all_ids
    state["early"] = cur_early_ids
    save_state(state)
    new_ids, ended_ids = diff_new_ended(prev_all, {"signals": cur_all_ids})
    # Group ended signals by stage prefix and show them inside Section 4 (no separate changelog).
    ended_by_stage = {"EARLY": [], "CONFIRMED": [], "VALIDATED": []}
    for _x in ended_ids:
        try:
            _t, _sig = _x.split("|", 1)
        except Exception:
            _sig = str(_x)
        if str(_sig).startswith("EARLY_"):
            ended_by_stage["EARLY"].append(_x)
        elif str(_sig).startswith("CONFIRMED_"):
            ended_by_stage["CONFIRMED"].append(_x)
        elif str(_sig).startswith("VALIDATED_"):
            ended_by_stage["VALIDATED"].append(_x)

    # Add explicit exit events (trailing giveback / neckline invalidation) with reason.
    # These are deterministic "ended today" cases even if the signal ID would also appear in ended_ids.
    try:
        exit_events = debug.get("exit_events", []) if isinstance(debug, dict) else []
        if isinstance(exit_events, list) and exit_events:
            for ev in exit_events:
                if not isinstance(ev, dict):
                    continue
                stg = str(ev.get("stage", "")).upper()
                tkr = str(ev.get("ticker", "")).strip()
                sig = str(ev.get("signal", "")).strip()
                reason = str(ev.get("reason", "")).strip()
                try:
                    pk = float(ev.get("peak_excess"))
                    cur = float(ev.get("cur_excess"))
                    extra = f" (peak={pk:.2f}ATR, now={cur:.2f}ATR)"
                except Exception:
                    extra = ""
                line = f"{tkr}|{sig} — {reason}{extra}"
                if stg.startswith("VAL"):
                    ended_by_stage["VALIDATED"].append(line)
                elif stg.startswith("CONF"):
                    ended_by_stage["CONFIRMED"].append(line)
    except Exception:
        pass
    new_early_ids, _ended_early_ids = diff_new_ended(prev_early, {"signals": cur_early_ids})
    new_set = set(new_early_ids)
    def mark_new(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            return df, df
        d = df.copy()
        d["_id"] = d["Ticker"].astype(str) + "|" + d["Signal"].astype(str)
        d_new = d[d["_id"].isin(new_set)].drop(columns=["_id"])
        d_old = d[~d["_id"].isin(new_set)].drop(columns=["_id"])
        return d_new, d_old
        # EARLY new/ongoing is diffed vs last run (pre-break proximity is transient).
    df_early = signals_to_df(early_sorted, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    df_early_new, df_early_old = mark_new(df_early)
    # CONFIRMED / VALIDATED new/ongoing is deterministic (based on breakout-day age), independent of whether the script ran.
    conf_new = [s for s in confirmed_sorted if getattr(s, "stage_status", None) == "NEW"]
    conf_old = [s for s in confirmed_sorted if getattr(s, "stage_status", None) == "ONGOING"]
    val_new = [s for s in validated_sorted if getattr(s, "stage_status", None) == "NEW"]
    val_old = [s for s in validated_sorted if getattr(s, "stage_status", None) == "ONGOING"]
    df_conf_new = signals_to_df(conf_new, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    df_conf_old = signals_to_df(conf_old, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    df_val_new = signals_to_df(val_new, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    df_val_old = signals_to_df(val_old, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
    # Assemble markdown
    md: List[str] = []
    md.append("# Daily Report\n")
    md.append(f"_Generated: **{header_time}** | Version: **{SCAN_VERSION}**_\n")
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
            f"<td style='padding-right:12px;'><img src='{_cb_img(vix_card)}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'></td>"
            f"<td><img src='{_cb_img(eur_card)}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'></td>"
            f"</tr></table>\n"
        )
    elif vix_card:
        md.append(f"<img src='{_cb_img(vix_card)}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'>\n")
    elif eur_card:
        md.append(f"<img src='{_cb_img(eur_card)}' width='{W}' style='width:{W}px;max-width:{W}px;height:auto;'>\n")
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
# ----------------------------
    # Signal engine health (diagnostics)
    # ----------------------------
    try:
        cand_total = int(debug.get("cand_total", 0))
        sig_total = int(debug.get("signals_total", 0))
        byp = debug.get("cand_by_pattern", {}) if isinstance(debug.get("cand_by_pattern", {}), dict) else {}
        top_pats = sorted([(k, int(v)) for k, v in byp.items()], key=lambda x: x[1], reverse=True)[:8]
        top_pats_str = ", ".join([f"{k}:{v}" for k, v in top_pats]) if top_pats else "None"
        md.append("### Signal engine health (diagnostics)\n")
        md.append(f"- Tickers scanned: **{int(debug.get('tickers_scanned', 0))}**; usable OHLCV: **{int(debug.get('tickers_usable', 0))}**\n")
        md.append(f"- Candidates found: **{cand_total}** (top patterns: {top_pats_str})\n")
        md.append(f"- Live signals: EARLY **{int(debug.get('signals_early', 0))}**, CONFIRMED **{int(debug.get('signals_conf', 0))}**, VALIDATED **{int(debug.get('signals_val', 0))}** (total {sig_total})\n")
        md.append(f"- Geometry restored today: HS/IHS **{int(debug.get('hs_restored', 0))}**, Band **{int(debug.get('band_restored', 0))}**\n")
        # Download diagnostic
        if _dl_diag.get('no_data', 0) > 0 or _dl_diag.get('short', 0) > 0:
            md.append(f"- Download: universe **{_dl_diag['universe']}** → downloaded **{_dl_diag['downloaded']}** → usable(≥80 bars) **{_dl_diag['usable']}** | short(<80) **{_dl_diag['short']}** | no data **{_dl_diag['no_data']}**\n")
            if _dl_diag.get('fail_by_exch'):
                md.append(f"  - No data by exchange: {', '.join(f'{k}: {v}/{_total_by_exch.get(k,0)}' for k, v in _dl_diag['fail_by_exch'])}\n")
            if _dl_diag.get('fail_by_country'):
                md.append(f"  - No data by country: {', '.join(f'{k}: {v}' for k, v in _dl_diag['fail_by_country'])}\n")
        # GC data layer diagnostic — load gc_state.json if available
        try:
            gc_state_path = DOCS_DIR / "gc_state.json"
            if gc_state_path.exists():
                import json as _json
                gc_state = _json.loads(gc_state_path.read_text(encoding="utf-8"))
                ec = gc_state.get("earnings_cache", {})
                # Filter to active tickers only (exclude inactive/below_min_mcap)
                ec = {k: v for k, v in ec.items()
                      if not v.get("inactive") and not v.get("below_min_mcap")}
                gc_total = len(ec)
                gc_rev = sum(1 for d in ec.values() if len(d.get("quarterly_revenue", [])) >= 4)
                gc_eps = sum(1 for d in ec.values() if any(e.get("eps_reported") for e in d.get("earnings_dates", [])))
                gc_catalyst = sum(1 for d in ec.values() if d.get("catalyst_events"))
                gc_blind = sum(1 for d in ec.values()
                    if len(d.get("quarterly_revenue", [])) < 4
                    and not any(e.get("eps_reported") for e in d.get("earnings_dates", []))
                    and not d.get("info", {}).get("revenue_growth"))

                # GC candidates: rev >= 20% YoY + >= 2 EPS beats
                import math as _math
                def _yoy(d):
                    wg = [r for r in d.get("quarterly_revenue", []) if r.get("revenue_yoy_growth") is not None]
                    if wg:
                        v = sorted(wg, key=lambda r: r["date"], reverse=True)[0]["revenue_yoy_growth"]
                        return float(v) if v is not None else None
                    ig = d.get("info", {}).get("revenue_growth")
                    if ig is not None:
                        try:
                            v = float(ig) * 100
                            return v if _math.isfinite(v) else None
                        except Exception: pass
                    return None
                def _beat_streak(d):
                    past = sorted([e for e in d.get("earnings_dates", []) if e.get("eps_reported") is not None],
                                  key=lambda e: e.get("date", ""), reverse=True)
                    streak = 0
                    for e in past:
                        s = e.get("eps_surprise_pct")
                        if s is None:
                            est = e.get("eps_estimate"); rep = e.get("eps_reported")
                            if est and rep and abs(float(est)) > 0.001:
                                s = (float(rep) / float(est) - 1.0) * 100
                        if s is not None and float(s) > 0: streak += 1
                        else: break
                    return streak

                def _rev_beat_streak(d):
                    """Consecutive quarters with BOTH EPS and revenue beat.
                    Revenue beat: consensus estimate if available, else YoY growth > 0
                    from quarterly_revenue (income statement) as proxy."""
                    past = sorted([e for e in d.get("earnings_dates", []) if e.get("eps_reported") is not None],
                                  key=lambda e: e.get("date", ""), reverse=True)
                    # Build YYYY-MM -> yoy lookup from income statement revenue
                    rev_yoy = {}
                    for qr in d.get("quarterly_revenue", []):
                        dt_ = (qr.get("date") or "")[:7]
                        g = qr.get("revenue_yoy_growth")
                        if dt_ and g is not None:
                            try: rev_yoy[dt_] = float(g)
                            except Exception: pass
                    def _rev_beat(e):
                        try:
                            r_est = e.get("revenue_estimate")
                            r_rep = e.get("revenue_reported")
                            if r_est is not None and r_rep is not None and float(r_est) > 0:
                                return float(r_rep) > float(r_est)
                        except Exception: pass
                        # Proxy: YoY > 0 from income stmt
                        d_ = (e.get("date") or "")[:7]
                        if d_ in rev_yoy: return rev_yoy[d_] > 0
                        try:
                            y, m = int(d_[:4]), int(d_[5:7])
                            for delta in [-1, -2, 1, 2]:
                                nm = m + delta; ny = y + (nm - 1) // 12; nm = ((nm - 1) % 12) + 1
                                key = f"{ny:04d}-{nm:02d}"
                                if key in rev_yoy: return rev_yoy[key] > 0
                        except Exception: pass
                        return False
                    streak = 0
                    for e in past:
                        rev_beat = _rev_beat(e)
                        s = e.get("eps_surprise_pct")
                        if s is None:
                            est = e.get("eps_estimate"); rep = e.get("eps_reported")
                            try:
                                if est and rep and abs(float(est)) > 0.001:
                                    s = (float(rep) / float(est) - 1.0) * 100
                            except Exception: pass
                        eps_beat = s is not None and float(s) > 0
                        if rev_beat and eps_beat: streak += 1
                        else: break
                    return streak

                # Star 2: BOTH EPS beat AND revenue beat >= 2 consecutive quarters
                # Note: revenue_beat_streak is 0 when yfinance lacks revenue estimates (common
                # outside US/EU), so gc_star2_eps_only shows how many are undercounted.
                gc_star2 = sum(1 for d in ec.values()
                               if _beat_streak(d) >= 2 and _rev_beat_streak(d) >= 2)
                gc_star2_eps_only = sum(1 for d in ec.values()
                               if _beat_streak(d) >= 2 and _rev_beat_streak(d) < 2)
                gc_catalyst = sum(1 for d in ec.values() if any(
                    e.get("ai_confirmed_massive") for e in d.get("catalyst_events", [])))
                # Star 3: Star 2 + last-Q rev >= 20% YoY (FY/TTM blocked) — moat via OpenAI in scan
                def _rev_source(d):
                    if d.get("quarterly_revenue"): return "quarterly"
                    if d.get("_rev_fallback"): return d["_rev_fallback"]
                    if d.get("info", {}).get("revenue_growth") is not None: return "info_fallback"
                    return "none"
                gc_layer3 = sum(1 for d in ec.values()
                    if (_yoy(d) or 0) >= 20.0
                    and _rev_source(d) not in ("info_fallback", "annual_estimated", "none"))
                gc_confirmed = sum(1 for d in ec.values()
                    if _beat_streak(d) >= 2
                    and _rev_beat_streak(d) >= 2
                    and (_yoy(d) or 0) >= 20.0
                    and _rev_source(d) not in ("info_fallback", "annual_estimated", "none"))
                gc_updated = gc_state.get("last_data_update", "unknown")[:10]

                # World vs EM + Korea breakdown — use imported path constants from universe.py
                world_count = em_count = 0
                try:
                    if MSCI_CSV.exists():
                        world_count = len(pd.read_csv(MSCI_CSV, dtype=str))
                    if MSCI_EM_CSV.exists():
                        em_count = len(pd.read_csv(MSCI_EM_CSV, dtype=str))
                except Exception: pass

                gc_fmp = sum(1 for d in ec.values() if d.get("data_source") == "fmp_fallback")

                md.append(f"- **GC Data Layer** (as of {gc_updated}): universe **{gc_total}** tickers")
                if world_count or em_count:
                    parts = []
                    if world_count: parts.append(f"World: {world_count}")
                    if em_count:    parts.append(f"EM: {em_count}")
                    md.append(f" ({' + '.join(parts)})")
                md.append(f" | rev data: **{gc_rev}** ({gc_rev*100//max(gc_total,1)}%) | EPS history: **{gc_eps}** ({gc_eps*100//max(gc_total,1)}%) | blind: **{gc_blind}** ({gc_blind*100//max(gc_total,1)}%)\n")
                # ── Data source breakdown for all 4 fields ────────────────────────
                # Columns:
                #   yf_ed  = yfinance earnings_dates HTML (M1/M3) — EPS reported + EPS estimate
                #            Works for ALL markets: US, EU, India, APAC numeric (KS/TW/T/SS/SZ etc.)
                #            yfinance scrapes Yahoo Finance Analysis page which has global analyst coverage
                #   yf_inc = yfinance quarterly_income_stmt (M4 + Phase A linkage)
                #            Revenue reported actuals for all markets with income statement data
                #   yf_fwd = yfinance quoteSummary earningsTrend — forward revenue estimates
                #            Same analyst coverage as yf_ed; fixed in v0.6.5 (crumb auth)
                #   fmp    = FMP /stable/earnings + alpha-batch
                #            Revenue estimate for US (primary); genuine gap-filler for others
                #   none   = genuinely missing (dead markets, no analyst coverage)
                _SRC_BUCKETS = ("yf_ed", "yf_inc", "yf_fwd", "fmp", "none")
                cov4 = {f"{field}_{s}": 0
                        for field in ("eps_rep","eps_est","rev_rep","rev_est")
                        for s in _SRC_BUCKETS}

                def _bucket_method(r_list):
                    """Map _method tag to source bucket for EPS reported."""
                    methods = [r.get("_method","") for r in r_list]
                    dom = max(set(methods), key=methods.count) if methods else ""
                    if "income_stmt" in dom: return "yf_inc"
                    return "yf_ed"  # earnings_dates, get_earnings_dates

                def _bucket_src_tag(src_tag):
                    """Map _*_source tag to source bucket."""
                    if not src_tag: return "yf_ed"  # untagged = from earnings_dates HTML
                    s = src_tag.lower()
                    if "fmp" in s: return "fmp"
                    if s in ("yf_income_stmt", "yf_inc"): return "yf_inc"
                    if s in ("yahoo_qs", "yf_fwd"): return "yf_fwd"
                    return "yf_ed"  # "yfinance" tag = earnings_dates HTML

                for d in ec.values():
                    past_ed = [r for r in d.get("earnings_dates",[]) if r.get("eps_reported") is not None]
                    if not past_ed: continue

                    # EPS reported — from earnings_dates HTML (82%) or income_stmt derived (17%)
                    cov4[f"eps_rep_{_bucket_method(past_ed)}"] += 1

                    # EPS estimate — yfinance earnings_dates HTML for ALL global markets (82%)
                    # Korea/Taiwan/Japan/India all have analyst consensus on Yahoo Finance
                    ee = [r for r in past_ed if r.get("eps_estimate") is not None]
                    if ee:
                        srcs = [r.get("_eps_est_source","yfinance") for r in ee]
                        dom = max(set(srcs), key=srcs.count)
                        cov4[f"eps_est_{_bucket_src_tag(dom)}"] += 1
                    else:
                        cov4["eps_est_none"] += 1

                    # Revenue reported — income_stmt linkage (Phase A, free, ~85% post v0.6.5)
                    rr = [r for r in past_ed if r.get("revenue_reported") is not None]
                    if rr:
                        srcs = [r.get("_rev_act_source","yfinance") for r in rr]
                        dom = max(set(srcs), key=srcs.count)
                        cov4[f"rev_rep_{_bucket_src_tag(dom)}"] += 1
                    else:
                        cov4["rev_rep_none"] += 1

                    # Revenue estimate — FMP currently (32%); earningsTrend (v0.6.5) adds ~50%
                    re_ = [r for r in d.get("earnings_dates",[]) if r.get("revenue_estimate") is not None]
                    if re_:
                        srcs = [r.get("_rev_est_source","yfinance") for r in re_]
                        dom = max(set(srcs), key=srcs.count)
                        cov4[f"rev_est_{_bucket_src_tag(dom)}"] += 1
                    else:
                        cov4["rev_est_none"] += 1

                # Also count forward_estimates coverage (earningsTrend, v0.6.5+)
                fwd_covered = sum(1 for d in ec.values()
                                  if d.get("forward_estimates") and not d.get("inactive") and not d.get("below_min_mcap"))

                tot = max(gc_total, 1)
                def _pct(n): return f"{n*100//tot}%"
                def _fmt(n): return f"**{n}** ({_pct(n)})"
                def _fmt4(field): return " | ".join(
                    _fmt(cov4[f"{field}_{s}"]) for s in _SRC_BUCKETS
                )

                # ── IMPORTANT: do NOT add \n inside md.append() for table rows.
                # "\n".join(md) already adds newlines between items. Adding \n inside
                # creates double-newlines which break the markdown `tables` extension,
                # causing pipe characters to render as raw text in the email.
                md.append(f"\n**Data source breakdown** (per ticker with EPS history, dominant source):\n")
                md.append(f"| Field | yf earn_dates | yf income_stmt | yf fwd_est | FMP | missing |")
                md.append(f"| :--- | ---: | ---: | ---: | ---: | ---: |")
                md.append(f"| EPS reported | {_fmt4('eps_rep')} |")
                md.append(f"| EPS estimate | {_fmt4('eps_est')} |")
                md.append(f"| Rev reported | {_fmt4('rev_rep')} |")
                md.append(f"| Rev estimate | {_fmt4('rev_est')} |")
                _fwd_note = f" · fwd_est covered: **{fwd_covered}** tickers" if fwd_covered else " · fwd_est: pending (earningsTrend v0.6.5)"
                md.append(f"\n_yf earn_dates=HTML table all markets · yf income_stmt=quarterly financials · yf fwd_est=earningsTrend API · FMP=paid gap-filler{_fwd_note}_\n")

                # ── Forward revenue estimate diagnostic ──────────────────────────────
                # For every ticker with a yfinance EPS estimate, show whether we have
                # a forward revenue estimate from yfinance (earningsTrend) or FMP,
                # or neither. This tells us the true residual value of FMP subscription.
                # Post v0.6.5: earningsTrend should cover same universe as EPS estimates.
                # Grouped by market suffix so gaps are actionable.
                _fwd_diag: dict = {}  # suffix → {n, has_yf_fwd, has_fmp_only, has_none}
                for _t, _d in ec.items():
                    if _d.get("inactive") or _d.get("below_min_mcap"):
                        continue
                    _ed = _d.get("earnings_dates", [])
                    # Only analyse tickers where yfinance gave us EPS estimates
                    if not any(_r.get("eps_estimate") and
                               (_r.get("_eps_est_source","yfinance") not in ("fmp","finnhub"))
                               for _r in _ed):
                        continue
                    _sfx = _t.rsplit(".",1)[-1] if "." in _t else "US"
                    if _sfx not in _fwd_diag:
                        _fwd_diag[_sfx] = {"n":0,"yf_fwd":0,"fmp_only":0,"none":0}
                    _fwd_diag[_sfx]["n"] += 1
                    # Check revenue estimate source
                    _rev_ests = [_r.get("_rev_est_source") for _r in _ed if _r.get("revenue_estimate")]
                    _has_yf_fwd = any(_s in ("yahoo_qs","yf_fwd") for _s in _rev_ests) or bool(_d.get("forward_estimates"))
                    _has_fmp    = any("fmp" in str(_s or "") for _s in _rev_ests)
                    if _has_yf_fwd:
                        _fwd_diag[_sfx]["yf_fwd"] += 1
                    elif _has_fmp:
                        _fwd_diag[_sfx]["fmp_only"] += 1
                    else:
                        _fwd_diag[_sfx]["none"] += 1

                # Sort by N descending, only show markets with ≥5 tickers
                _diag_rows = sorted(
                    [(s, d) for s, d in _fwd_diag.items() if d["n"] >= 5],
                    key=lambda x: -x[1]["n"]
                )
                _diag_flags = {
                    "US":"🇺🇸","TO":"🇨🇦","L":"🇬🇧","DE":"🇩🇪","PA":"🇫🇷","NS":"🇮🇳",
                    "TW":"🇹🇼","KS":"🇰🇷","T":"🇯🇵","HK":"🇭🇰","SS":"🇨🇳","SZ":"🇨🇳",
                    "SA":"🇧🇷","AX":"🇦🇺","ST":"🇸🇪","MX":"🇲🇽","IS":"🇹🇷","JK":"🇮🇩",
                    "SR":"🇸🇦","KW":"🇰🇼","SW":"🇨🇭","WA":"🇵🇱","MI":"🇮🇹","MC":"🇪🇸",
                }

                if _diag_rows:
                    _yf_tot  = sum(d["yf_fwd"]   for _, d in _diag_rows)
                    _fmp_tot = sum(d["fmp_only"]  for _, d in _diag_rows)
                    _non_tot = sum(d["none"]       for _, d in _diag_rows)
                    _all_tot = sum(d["n"]          for _, d in _diag_rows)
                    md.append(f"\n**Forward revenue estimate coverage** (tickers where yfinance provides EPS estimate):\n")
                    md.append(f"| Market | N | yf earningsTrend | FMP only | no rev est |")
                    md.append(f"| :--- | ---: | ---: | ---: | ---: |")
                    for _sfx, _dd in _diag_rows:
                        _flag = _diag_flags.get(_sfx, f".{_sfx}")
                        def _dp(x): return f"**{x}** ({x*100//_dd['n']}%)" if x else "–"
                        md.append(f"| {_flag} .{_sfx} | {_dd['n']} | {_dp(_dd['yf_fwd'])} | {_dp(_dd['fmp_only'])} | {_dp(_dd['none'])} |")
                    md.append(f"| **Total** | **{_all_tot}** | **{_yf_tot}** ({_yf_tot*100//_all_tot}%) | **{_fmp_tot}** ({_fmp_tot*100//_all_tot}%) | **{_non_tot}** ({_non_tot*100//_all_tot}%) |")
                    md.append(f"\n_yf earningsTrend: free, covers all markets with analyst consensus · FMP only: paid, fills where earningsTrend unavailable · no rev est: no analyst revenue consensus (small-cap, local-only)_\n")


                # ── Per-country estimate coverage (full breakdown, all markets) ──
                # Answers: which countries have EPS-only, Rev-only, or no estimates at all?
                from collections import defaultdict as _dd2
                _EXCH_FLAGS = {
                    "US": "🇺🇸 US",        "TO": "🇨🇦 Canada",     "L":  "🇬🇧 UK",
                    "DE": "🇩🇪 Germany",    "PA": "🇫🇷 France",      "AS": "🇳🇱 Netherlands",
                    "MI": "🇮🇹 Italy",      "MC": "🇪🇸 Spain",       "SW": "🇨🇭 Switzerland",
                    "ST": "🇸🇪 Sweden",     "OL": "🇳🇴 Norway",      "HE": "🇫🇮 Finland",
                    "CO": "🇩🇰 Denmark",    "AT": "🇬🇷 Greece",      "VI": "🇦🇹 Austria",
                    "IR": "🇮🇪 Ireland",    "LS": "🇵🇹 Portugal",    "WA": "🇵🇱 Poland",
                    "BD": "🇭🇺 Hungary",    "PR": "🇨🇿 Czech Rep.",   "T":  "🇯🇵 Japan",
                    "HK": "🇭🇰 Hong Kong",  "KS": "🇰🇷 S. Korea",    "TW": "🇹🇼 Taiwan",
                    "SI": "🇸🇬 Singapore",  "AX": "🇦🇺 Australia",   "NS": "🇮🇳 India",
                    "SA": "🇧🇷 Brazil/SA",  "JO": "🇿🇦 S. Africa",   "MX": "🇲🇽 Mexico",
                    "JK": "🇮🇩 Indonesia",  "BK": "🇹🇭 Thailand",    "KL": "🇲🇾 Malaysia",
                    "IS": "🇹🇷 Turkey",     "TA": "🇮🇱 Israel",      "AE": "🇦🇪 UAE",
                    "AD": "🇦🇪 UAE-AD",     "DU": "🇦🇪 UAE-DU",     "SN": "🇨🇱 Chile",
                    "KQ": "🇰🇷 Korea(KQ)",  "TWO":"🇹🇼 Taiwan(OTC)", "BO": "🇮🇳 India(BSE)",
                    "BR": "🇧🇪 Belgium",
                }
                _ct: dict = _dd2(lambda: {"n":0,"er":0,"ee":0,"rr":0,"re":0,"a4":0,"blind":0,"eps_o":0,"rev_o":0})
                for ticker, d in ec.items():
                    exch = ticker.split(".")[-1] if "." in ticker else "US"
                    c = _EXCH_FLAGS.get(exch, f".{exch}")
                    past = [r for r in d.get("earnings_dates",[]) if r.get("eps_reported") is not None]
                    er = len(past) > 0
                    ee = any(r.get("eps_estimate") is not None for r in past)
                    rr = any(r.get("revenue_reported") is not None for r in past)
                    re = any(r.get("revenue_estimate") is not None for r in past)
                    _ct[c]["n"] += 1
                    if er: _ct[c]["er"] += 1
                    if ee: _ct[c]["ee"] += 1
                    if rr: _ct[c]["rr"] += 1
                    if re: _ct[c]["re"] += 1
                    if er and ee and rr and re: _ct[c]["a4"] += 1
                    if er and not ee and not re: _ct[c]["blind"] += 1
                    if er and ee and not re: _ct[c]["eps_o"] += 1
                    if er and re and not ee: _ct[c]["rev_o"] += 1

                md.append(f"\n**Per-country estimate coverage** (N = tickers | EPS est only = has EPS est but no Rev est | No est = both missing):\n")
                md.append(f"| Country | N | EPS rep | EPS est | Rev rep | Rev est | All-4 | EPS est only | Rev est only | No est ⚠ |")
                md.append(f"| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                _ctot2 = {"n":0,"er":0,"ee":0,"rr":0,"re":0,"a4":0,"blind":0,"eps_o":0,"rev_o":0}
                for c, d in sorted(_ct.items(), key=lambda x: -x[1]["n"]):
                    cn = d["n"]
                    if cn < 1: continue
                    def _cp(v): return f"**{v}** ({v*100//cn}%)"
                    blind_cell = f"**{d['blind']}** ⚠" if d["blind"] > 5 else str(d["blind"])
                    md.append(
                        f"| {c} | {cn} | {_cp(d['er'])} | {_cp(d['ee'])} | {_cp(d['rr'])} | {_cp(d['re'])} "
                        f"| {d['a4']} | {d['eps_o'] or '–'} | {d['rev_o'] or '–'} | {blind_cell} |"
                    )
                    for k in _ctot2: _ctot2[k] += d[k]
                tn2 = max(_ctot2["n"], 1)
                md.append(
                    f"| **TOTAL** | **{_ctot2['n']}** | **{_ctot2['er']}** ({_ctot2['er']*100//tn2}%) "
                    f"| **{_ctot2['ee']}** ({_ctot2['ee']*100//tn2}%) "
                    f"| **{_ctot2['rr']}** ({_ctot2['rr']*100//tn2}%) "
                    f"| **{_ctot2['re']}** ({_ctot2['re']*100//tn2}%) "
                    f"| **{_ctot2['a4']}** | {_ctot2['eps_o']} | {_ctot2['rev_o']} | **{_ctot2['blind']}** ⚠ |"
                )
                md.append(f"\n**Estimate coverage gaps** (tickers with EPS reported):\n")
                md.append(f"| Gap type | Count | Action |")
                md.append(f"| :--- | ---: | :--- |")
                md.append(f"| No estimates at all (EPS + Rev missing) | **{_ctot2['blind']}** | IC / FMP non-US expansion — top: India ({_ct.get('🇮🇳 India',{}).get('blind',0)}), Brazil/SA ({_ct.get('🇧🇷 Brazil/SA',{}).get('blind',0)}), Turkey ({_ct.get('🇹🇷 Turkey',{}).get('blind',0)}), Indonesia ({_ct.get('🇮🇩 Indonesia',{}).get('blind',0)}), Thailand ({_ct.get('🇹🇭 Thailand',{}).get('blind',0)}) |")
                md.append(f"| EPS estimate only (Rev est missing) | {_ctot2['eps_o']} | FMP /stable/earnings; IC for UK/EU |")
                md.append(f"| Rev estimate only (EPS est missing) | {_ctot2['rev_o']} | Check FMP symbol mapping |")

                md.append(
                    f"\n_Cascade: yfinance → investing.com → FMP → Finnhub. "
                    f"Without consensus estimate, revenue beat/miss cannot be computed._\n"
                )
                md.append(f"  - ★★ Star 2 (dual EPS **and** Rev beat ≥2Q): **{gc_star2}**"
                          f"  (+{gc_star2_eps_only} data-gap single-beat fallback)")
                if gc_catalyst > 0:
                    md.append(f" | +**{gc_catalyst}** AI-confirmed catalysts (Star 2 via catalyst path)")
                md.append(f"\n")
                md.append(f"  - ★★★ Star 3 (Star 2 + last-Q rev ≥20% YoY): **{gc_confirmed}**"
                          f"  — moat confirmed per ticker via OpenAI\n")
                md.append(f"  - Last-Q rev ≥20% YoY pool (quarterly source only): **{gc_layer3}** tickers\n")
                if gc_fmp > 0:
                    md.append(f"  - FMP fallback: **{gc_fmp}** tickers recovered via Financial Modeling Prep (yfinance had no data)\n")
                else:
                    md.append(f"  - Data source: yfinance only (FMP_API_KEY not set or not needed)\n")
        except Exception:
            pass
        md.append("\n")
    except Exception:
        pass
    # Focus tickers deep-dive (always shown) with full HS/IHS diagnostics + charts
    md.append("### Focus tickers deep-dive\n")
    md.append(f"_Full diagnostics + validation gate history for: {', '.join(FOCUS_TICKERS)}_\n\n")
    for ft in FOCUS_TICKERS:
        try:
            df_ft = ohlcv.get(ft)
            if df_ft is None or df_ft.empty:
                md.append(f"**{ft}** — no data\n\n")
                continue
            # Align all focus diagnostics and charts to the same detector window (trading-bar lookback).
            d_local = df_ft.dropna(subset=["Open","High","Low","Close"]).copy()
            d_local = d_local.tail(LOOKBACK_DAYS).copy()
            d_local = _latest_completed_close_df(d_local)
            if d_local.empty or len(d_local) < 5:
                md.append(f"**{ft}** — insufficient bars\n\n")
                continue
            info = _debug_gates_for_ticker(ft, df_ft, state=state, max_candidates=12)
            best = info.get("Best") or {}
            nm = company_name_for_ticker(ft)
            nm_disp = (NAME_OVERRIDES.get(ft) or nm or ft).upper()
            md.append(f"**{nm_disp} ({ft})**\n")
            top_list = info.get("Top") or []
            hs_seen = any((isinstance(r, dict) and r.get("pattern") in ("HS_TOP", "IHS")) for r in top_list)
            md.append(f"- HS/IHS detected today: **{'YES' if hs_seen else 'NO'}**\n")
            # Always compute HS/IHS gate diagnostics for focus tickers (even if a HS/IHS was detected).
            exp_top: Dict[str, Any] = {}
            exp_inv: Dict[str, Any] = {}
            try:
                _ = detect_hs_top(df_ft, explain=exp_top)
            except Exception:
                pass
            try:
                _ = detect_inverse_hs(df_ft, explain=exp_inv)
            except Exception:
                pass
            def _fmt(exp: Dict[str, Any]) -> str:
                items = []
                for k, v in exp.items():
                    if k in ("highs", "lows") or str(k).startswith("_"):
                        continue
                    if isinstance(v, (int, float)) and v:
                        items.append((k, int(v)))
                items.sort(key=lambda x: x[1], reverse=True)
                top = items[:10]
                return ", ".join([f"{k}:{v}" for k, v in top]) if top else "None"
            md.append(f"- HS_TOP reject summary: {_fmt(exp_top)}\n")
            md.append(f"- IHS reject summary: {_fmt(exp_inv)}\n")
            # If HS/IHS was rejected by guardrails, show the last rejected geometry for transparency (focus tickers).
            try:
                lr = exp_top.get("_last_reject_geom") if isinstance(exp_top, dict) else None
                if isinstance(lr, dict) and isinstance(lr.get("geom"), dict):
                    g = lr["geom"]
                    md.append("- Last rejected HS geometry (guardrail):\n")
                    md.append(f"  - LS/H/RS (ts): LS={lr.get('LS_t')}, H={lr.get('H_t')}, RS={lr.get('RS_t')}\n")
                    md.append(f"  - 1) H global in [LS..RS]: {'YES' if g.get('head_is_global_span') else 'NO'} (H_Close={g.get('head_close')}, span_extreme={g.get('head_span_extreme_close')}, extreme_i={g.get('head_span_arg_i')})\n")
                    md.append(f"  - 2) LS local (±{HS_LOCAL_WINDOW}): {'YES' if g.get('ls_local_extreme') else 'NO'} | RS local: {'YES' if g.get('rs_local_extreme') else 'NO'}\n")
                    md.append(f"  - 3) Symmetry ≥ {HS_SYMMETRY_MIN_RATIO:.2f}: {'YES' if g.get('symmetry_ok') else 'NO'} (dL={g.get('dL')}, dR={g.get('dR')}, ratio={g.get('symmetry_ratio')})\n")
                    md.append(f"  - 4) Valley ≥ {HS_VALLEY_ATR_MULT:.1f}×ATR: {'YES' if g.get('valley_ok') else 'NO'} (left={g.get('valley_left_depth')}, right={g.get('valley_right_depth')}, thr={g.get('valley_thr')}, ATR={g.get('atr_head')})\n")
                    md.append(f"  - 5) Shoulder→Valley ≥ {HS_SHOULDER_VALLEY_ATR_MULT:.1f}×ATR: {'YES' if g.get('shoulder_valley_ok') else 'NO'} (left={g.get('shoulder_valley_left_depth')}, right={g.get('shoulder_valley_right_depth')}, thr={g.get('shoulder_valley_thr')})\n")
                    md.append(f"  - 6) Span in [{HS_MIN_BARS}..{HS_MAX_BARS}] bars: {'YES' if g.get('span_ok') else 'NO'} (span={g.get('span_bars')})\n")
                    md.append(f"  - 7) Sidebars ≥ {HS_MIN_SIDE_BARS} bars: {'YES' if g.get('sidebars_ok') else 'NO'} (dL={g.get('dL')}, dR={g.get('dR')})\n")
                # If no last rejected geometry was captured (e.g., rejection happened before guardrails),
                # produce a best-effort diagnostic geometry so focus tickers still show the deterministic checks.
                try:
                    lr2 = exp_top.get("_last_reject_geom") if isinstance(exp_top, dict) else None
                    if not (isinstance(lr2, dict) and isinstance(lr2.get("geom"), dict)):
                        d_det2 = df_ft.tail(LOOKBACK_DAYS).dropna(subset=["Open","High","Low","Close"]).copy()
                        d_det2 = _latest_completed_close_df(d_det2)
                        if len(d_det2) >= 60:
                            c2 = d_det2["Close"].astype(float).values
                            head_val2 = float(np.nanmax(c2))
                            head_idxs2 = np.where(np.isclose(c2, head_val2, rtol=0.0, atol=1e-8))[0]
                            p2_dbg = int(head_idxs2[-1]) if len(head_idxs2) else int(np.nanargmax(c2))
                            head_zone2 = int(HS_MIN_SIDE_BARS)
                            # Close-peak candidates for shoulders (same as HS_TOP)
                            hi2 = _swing_highs_on_close(d_det2, window=5, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
                            L_lo2 = max(0, p2_dbg - int(HS_MAX_BARS))
                            L_hi2 = max(L_lo2, p2_dbg - head_zone2)
                            R_lo2 = min(len(d_det2), p2_dbg + 1 + head_zone2)
                            R_hi2 = min(len(d_det2), p2_dbg + 1 + int(HS_MAX_BARS))
                            left2 = [int(i) for i in hi2 if L_lo2 <= int(i) < L_hi2]
                            right2 = [int(i) for i in hi2 if R_lo2 <= int(i) < R_hi2]
                            left2_sorted = sorted(left2, key=lambda i: (float(d_det2["Close"].iloc[i]), int(i)), reverse=True)
                            right2_sorted = sorted(right2, key=lambda i: (float(d_det2["Close"].iloc[i]), -int(i)), reverse=True)
                            md.append(f"- HS diagnostic head anchor: H={_iso_ts(d_det2.index[p2_dbg])} (i={p2_dbg}) | Close={float(d_det2['Close'].iloc[p2_dbg]):.2f}\n")
                            if left2_sorted:
                                md.append("  - Left shoulder candidates (top 5): " + ", ".join([f"{_iso_ts(d_det2.index[i])}({float(d_det2['Close'].iloc[i]):.2f})" for i in left2_sorted[:5]]) + "\n")
                            else:
                                md.append("  - Left shoulder candidates: NONE (after head-zone exclusion)\n")
                            if right2_sorted:
                                md.append("  - Right shoulder candidates (top 5): " + ", ".join([f"{_iso_ts(d_det2.index[i])}({float(d_det2['Close'].iloc[i]):.2f})" for i in right2_sorted[:5]]) + "\n")
                            else:
                                md.append("  - Right shoulder candidates: NONE (after head-zone exclusion)\n")
                            if left2_sorted and right2_sorted:
                                p1_dbg = int(left2_sorted[0])
                                p3_dbg = int(right2_sorted[0])
                                geom2 = _hs_geometry_diagnostics(
                                    d_det2, p1_dbg, p2_dbg, p3_dbg,
                                    inverse=False, local_window=HS_LOCAL_WINDOW,
                                    symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO,
                                    valley_atr_mult=HS_VALLEY_ATR_MULT
                                )
                                md.append("- Best attempted HS geometry (diagnostic):\n")
                                md.append(f"  - LS/H/RS (ts): LS={_iso_ts(d_det2.index[p1_dbg])}, H={_iso_ts(d_det2.index[p2_dbg])}, RS={_iso_ts(d_det2.index[p3_dbg])}\n")
                                md.append("  - Geometry checks (deterministic):\n")
                                md.append(f"    - 1) H is absolute extreme Close in [LS..RS]: {'YES' if geom2.get('head_is_global_span') else 'NO'} (H_Close={geom2.get('head_close'):.2f}, span_extreme={geom2.get('head_span_extreme_close'):.2f}, extreme_i={geom2.get('head_span_arg_i')})\n")
                                md.append(f"    - 2) LS local extreme (±{HS_LOCAL_WINDOW}): {'YES' if geom2.get('ls_local_extreme') else 'NO'} | RS local extreme: {'YES' if geom2.get('rs_local_extreme') else 'NO'}\n")
                                md.append(f"    - 3) Symmetry ratio ≥ {HS_SYMMETRY_MIN_RATIO:.2f}: {'YES' if geom2.get('symmetry_ok') else 'NO'} (dL={geom2.get('dL')}, dR={geom2.get('dR')}, ratio={geom2.get('symmetry_ratio'):.2f})\n")
                                md.append(f"    - 4) Valley depth ≥ {HS_VALLEY_ATR_MULT:.1f}×ATR(head): {'YES' if geom2.get('valley_ok') else 'NO'} (left={geom2.get('valley_left_depth'):.2f}, right={geom2.get('valley_right_depth'):.2f}, thr={geom2.get('valley_thr'):.2f}, ATR={geom2.get('atr_head'):.2f})\n")
                                md.append(f"    - 5) Shoulder→Valley ≥ {HS_SHOULDER_VALLEY_ATR_MULT:.1f}×ATR(head): {'YES' if geom2.get('shoulder_valley_ok') else 'NO'} (left={geom2.get('shoulder_valley_left_depth'):.2f}, right={geom2.get('shoulder_valley_right_depth'):.2f}, thr={geom2.get('shoulder_valley_thr'):.2f})\n")
                                md.append(f"    - 6) Span bars in [{HS_MIN_BARS}..{HS_MAX_BARS}]: {'YES' if geom2.get('span_ok') else 'NO'} (span={geom2.get('span_bars')})\n")
                                md.append(f"    - 7) Sidebars ≥ {HS_MIN_SIDE_BARS} bars: {'YES' if geom2.get('sidebars_ok') else 'NO'} (dL={geom2.get('dL')}, dR={geom2.get('dR')})\n")
                                md.append(f"    - 8) Reaction pivots local (±{HS_LOCAL_WINDOW}): {'YES' if geom2.get('react_local_ok') else 'NO'} (p1={geom2.get('react_i1')}, p2={geom2.get('react_i2')}, ok1={geom2.get('react1_local')}, ok2={geom2.get('react2_local')})\n")
                                md.append(f"    - 9) Neckline angle within ±{HS_NECKLINE_MAX_ANGLE_DEG:.1f}° (abs): {'YES' if geom2.get('neckline_angle_ok') else 'NO'} (angle={geom2.get('neckline_angle_deg'):.2f}°)\n")
                except Exception:
                    pass
            except Exception:
                pass
            if "highs" in exp_top or "lows" in exp_top:
                md.append(f"- Swings (HS_TOP): highs={exp_top.get('highs','?')} lows={exp_top.get('lows','?')}\n")
            if "highs" in exp_inv or "lows" in exp_inv:
                md.append(f"- Swings (IHS): highs={exp_inv.get('highs','?')} lows={exp_inv.get('lows','?')}\n")
            # NU-specific diagnostics: why Jan 5 is / is not a swing high + whether HS_MIN_BARS is forcing LS backward
            if str(ft).upper() == "NU":
                try:
                    diagc = _diagnose_swing_high_close(df_ft, "2026-01-05", window=5, prominence_atr_mult=0.5)
                    md.append(f"- NU swing-high check (Close-based) @ 2026-01-05: ok={diagc.get('ok')} | reason={diagc.get('reason')}\n")
                    if diagc.get("ts_hit"):
                        md.append(
                            f"  - hit={diagc.get('ts_hit')} (i={diagc.get('i')}) | Close={diagc.get('close_i'):.2f} | "
                            f"cwin_max={diagc.get('cwin_max'):.2f} (count={diagc.get('cwin_max_count')}) | "
                            f"prom={diagc.get('prominence'):.2f} vs thr={diagc.get('prom_thresh'):.2f} (ATR={diagc.get('atr_i'):.2f})\n"
                        )
                    diagh = _diagnose_swing_high(df_ft, "2026-01-05", window=3, prominence_atr_mult=0.5)
                    md.append(f"- NU swing-high check (High-based) @ 2026-01-05: ok={diagh.get('ok')} | reason={diagh.get('reason')}\n")
                    if diagh.get("ts_hit"):
                        md.append(
                            f"  - hit={diagh.get('ts_hit')} (i={diagh.get('i')}) | High={diagh.get('hi_i'):.2f} | "
                            f"hwin_max={diagh.get('hwin_max'):.2f} (count={diagh.get('hwin_max_count')}) | "
                            f"prom={diagh.get('prominence'):.2f} vs thr={diagh.get('prom_thresh'):.2f} (ATR={diagh.get('atr_i'):.2f})\n"
                        )
                except Exception:
                    pass
                # NU: show the actual left/right shoulder candidate peaks around the head (CLOSE-based) and why Jan 5 might lose.
                try:
                    d_det = df_ft.tail(LOOKBACK_DAYS).dropna(subset=["Open","High","Low","Close"]).copy()
                    d_det = _latest_completed_close_df(d_det)
                    if len(d_det) > 20 and isinstance(d_det.index, pd.DatetimeIndex):
                        c_det = d_det["Close"].astype(float)
                        # Same pivot definition as HS_TOP: Close swing highs with window=5
                        hi_det = _swing_highs_on_close(d_det, window=5, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
                        # Head = most recent global max close
                        head_val = float(np.nanmax(c_det.values))
                        head_idxs = np.where(np.isclose(c_det.values.astype(float), head_val, rtol=0.0, atol=1e-8))[0]
                        p2_det = int(head_idxs[-1]) if len(head_idxs) else int(np.nanargmax(c_det.values))
                        head_zone = int(HS_MIN_SIDE_BARS)
                        L_lo = max(0, p2_det - int(HS_MAX_BARS))
                        L_hi = max(L_lo, p2_det - head_zone)
                        R_lo = min(len(c_det), p2_det + 1 + head_zone)
                        R_hi = min(len(c_det), p2_det + 1 + int(HS_MAX_BARS))
                        left = [int(i) for i in hi_det if L_lo <= int(i) < L_hi]
                        right = [int(i) for i in hi_det if R_lo <= int(i) < R_hi]
                        left_sorted = sorted(left, key=lambda i: (float(c_det.iloc[i]), int(i)), reverse=True)
                        right_sorted = sorted(right, key=lambda i: (-float(c_det.iloc[i]), int(i)))
                        md.append(f"- NU HS_TOP head anchor (detector window): H={d_det.index[p2_det].date()} (i={p2_det}) | Close={float(c_det.iloc[p2_det]):.2f} | head_zone=±{head_zone} bars\n")
                        # Show top left candidates
                        if left_sorted:
                            md.append("- NU left-shoulder CLOSE-peak candidates (top 6):\n")
                            for k,i in enumerate(left_sorted[:6]):
                                md.append(f"  - L{k+1}: {d_det.index[i].date()} (i={i}) Close={float(c_det.iloc[i]):.2f}\n")
                        else:
                            md.append("- NU left-shoulder candidates: NONE (after head-zone exclusion)\n")
                        # Explicitly check Jan 5 membership
                        ts_j5 = pd.to_datetime('2026-01-05')
                        if ts_j5 in d_det.index:
                            i_j5 = int(d_det.index.get_loc(ts_j5))
                            in_left = (i_j5 in left)
                            md.append(f"- NU Jan 5 index in detector window: i={i_j5} Close={float(c_det.iloc[i_j5]):.2f} | in_left_candidates={in_left}\n")
                            if in_left and left_sorted:
                                top_i = left_sorted[0]
                                md.append(f"  - Left-candidate winner by CLOSE: {d_det.index[top_i].date()} (i={top_i}) Close={float(c_det.iloc[top_i]):.2f}\n")
                                md.append(f"  - Jan 5 loses only if another LEFT candidate has higher CLOSE outside the head-zone.\n")
                        else:
                            md.append("- NU Jan 5 not found in detector window index (data mismatch / missing bar)\n")
                except Exception:
                    pass
                try:
                    dr = exp_top.get("_dur_reject_best")
                    if isinstance(dr, dict):
                        md.append(
                            f"- NU duration gate (HS_MIN_BARS={dr.get('HS_MIN_BARS')}): most-recent rejected snapped candidate "
                            f"LS={dr.get('p1_t')} → RS={dr.get('p3_t')} span={dr.get('span')} bars (dL={dr.get('dL')}, dR={dr.get('dR')}, ratio={dr.get('ratio'):.2f})\n"
                        )
                except Exception:
                    pass
            if best and best.get("pattern"):
                patt = str(best.get("pattern", ""))
                direc = str(best.get("dir", ""))
                dist = float(best.get("distATR", float("nan")))
                price_ok = bool(best.get("price_ok"))
                clv_ok = bool(best.get("clv_ok"))
                vol_ok = bool(best.get("vol_ok"))
                hs_lag = str(best.get("hs_lag", ""))
                lvl = float(best.get("level", 0.0)) if best.get("level") is not None else float("nan")
                meta = best.get("meta")
                md.append(f"- Best candidate: **{patt} / {direc}** | Dist(ATR) **{dist:+.2f}** | Gates: Price **{'Y' if price_ok else 'N'}**, CLV **{'Y' if clv_ok else 'N'}**, Vol **{'Y' if vol_ok else 'N'}** | HS lag **{hs_lag}**\n")
                try:
                    clv_val = info.get('CLV', None)
                    volr_val = info.get('VolRatio', None)
                    if clv_val is not None or volr_val is not None:
                        md.append(f"  - Gate inputs: CLV={clv_val} | VolRatio={volr_val}\n")
                    try:
                        meta2 = best.get("meta") if isinstance(best, dict) else None
                        # Reindex geometry points onto the current detector window (prevents stale iloc indices)
                        if isinstance(meta2, dict):
                            meta2 = _reindex_meta_to_df(meta2, d_local) or meta2
                        pts = meta2.get("points") if isinstance(meta2, dict) else None
                        if isinstance(pts, list):
                            def _pt(lbl):
                                for p in pts:
                                    if isinstance(p, dict) and str(p.get("label","")) == lbl:
                                        return p
                                return None
                            pLS = _pt("LS"); pH = _pt("H"); pRS = _pt("RS")
                            if pLS and pH and pRS:
                                ls_i = int(pLS.get("i")); h_i = int(pH.get("i")); rs_i = int(pRS.get("i"))
                                md.append(f"  - LS/H/RS geometry (idx): LS={ls_i}, H={h_i}, RS={rs_i}\n")
                                md.append(f"  - LS/H/RS geometry (ts): LS={pLS.get('t')}, H={pH.get('t')}, RS={pRS.get('t')}\\n")
                                try:
                                    ls_p = _safe_float(pLS.get("p")); h_p = _safe_float(pH.get("p")); rs_p = _safe_float(pRS.get("p"))
                                    ls_c = _safe_float(d_local["Close"].iloc[ls_i]); h_c = _safe_float(d_local["Close"].iloc[h_i]); rs_c = _safe_float(d_local["Close"].iloc[rs_i])
                                    if np.isfinite(ls_p) and np.isfinite(ls_c) and np.isfinite(h_p) and np.isfinite(h_c) and np.isfinite(rs_p) and np.isfinite(rs_c):
                                        md.append(f"  - Point alignment (p vs Close): LSΔ={ls_p-ls_c:+.4f}, HΔ={h_p-h_c:+.4f}, RSΔ={rs_p-rs_c:+.4f}\\n")
                                except Exception:
                                    pass
                                # Deterministic geometry checks (requested)
                                try:
                                    geom_chk = _hs_geometry_diagnostics(d_local, ls_i, h_i, rs_i, inverse=(patt == "IHS"),
                                                                        local_window=HS_LOCAL_WINDOW,
                                                                        symmetry_min_ratio=HS_SYMMETRY_MIN_RATIO,
                                                                        valley_atr_mult=HS_VALLEY_ATR_MULT)
                                    md.append("  - Geometry checks (deterministic):\n")
                                    md.append(f"    - 1) H is absolute extreme Close in [LS..RS]: {'YES' if geom_chk.get('head_is_global_span') else 'NO'} (H_Close={geom_chk.get('head_close'):.2f}, span_extreme={geom_chk.get('head_span_extreme_close'):.2f}, extreme_i={geom_chk.get('head_span_arg_i')})\n")
                                    md.append(f"    - 2) LS local extreme (±{HS_LOCAL_WINDOW} bars): {'YES' if geom_chk.get('ls_local_extreme') else 'NO'} | RS local extreme: {'YES' if geom_chk.get('rs_local_extreme') else 'NO'}\n")
                                    md.append(f"    - 3) Symmetry ratio min(dL,dR)/max(dL,dR) ≥ {HS_SYMMETRY_MIN_RATIO:.2f}: {'YES' if geom_chk.get('symmetry_ok') else 'NO'} (dL={geom_chk.get('dL')}, dR={geom_chk.get('dR')}, ratio={geom_chk.get('symmetry_ratio'):.2f})\n")
                                    md.append(f"    - 4) Valley depth ≥ {HS_VALLEY_ATR_MULT:.1f}×ATR(head): {'YES' if geom_chk.get('valley_ok') else 'NO'} (left={geom_chk.get('valley_left_depth'):.2f}, right={geom_chk.get('valley_right_depth'):.2f}, thr={geom_chk.get('valley_thr'):.2f}, ATR={geom_chk.get('atr_head'):.2f})\n")
                                    md.append(f"    - 5) Shoulder→Valley ≥ {HS_SHOULDER_VALLEY_ATR_MULT:.1f}×ATR(head): {'YES' if geom_chk.get('shoulder_valley_ok') else 'NO'} (left={geom_chk.get('shoulder_valley_left_depth'):.2f}, right={geom_chk.get('shoulder_valley_right_depth'):.2f}, thr={geom_chk.get('shoulder_valley_thr'):.2f})\n")
                                    md.append(f"    - 6) Span bars in [{HS_MIN_BARS}..{HS_MAX_BARS}]: {'YES' if geom_chk.get('span_ok') else 'NO'} (span={geom_chk.get('span_bars')})\n")
                                    md.append(f"    - 7) Sidebars ≥ {HS_MIN_SIDE_BARS} bars: {'YES' if geom_chk.get('sidebars_ok') else 'NO'} (dL={geom_chk.get('dL')}, dR={geom_chk.get('dR')})\n")
                                    md.append(f"    - 8) Reaction pivots local (±{HS_LOCAL_WINDOW}): {'YES' if geom_chk.get('react_local_ok') else 'NO'} (p1={geom_chk.get('react_i1')}, p2={geom_chk.get('react_i2')}, ok1={geom_chk.get('react1_local')}, ok2={geom_chk.get('react2_local')})\n")
                                    md.append(f"    - 9) Neckline angle within ±{HS_NECKLINE_MAX_ANGLE_DEG:.1f}° (abs): {'YES' if geom_chk.get('neckline_angle_ok') else 'NO'} (angle={geom_chk.get('neckline_angle_deg'):.2f}°)\n")
                                except Exception:
                                    pass
                                # Use the SAME detector window (tail LOOKBACK_DAYS), so meta indices (iloc) always align.
                                d_local = df_ft.tail(LOOKBACK_DAYS).dropna(subset=["Open","High","Low","Close"]).copy()
                                d_local = _latest_completed_close_df(d_local)
                                # Sanity: head should be the MAX close in the detector window (HS_TOP) / MIN close (IHS).
                                try:
                                    close_det = d_local["Close"].astype(float)
                                    if patt == "HS_TOP":
                                        mxv = float(np.nanmax(close_det.values))
                                        mxi = int(np.nanargmax(close_det.values))
                                        hv = float(close_det.iloc[int(h_i)]) if 0 <= int(h_i) < len(close_det) else float("nan")
                                        md.append(f"  - Detector window maxClose: {mxv:.2f} at {d_local.index[mxi]} | H_Close={hv:.2f}\n")
                                    elif patt == "IHS":
                                        mnv = float(np.nanmin(close_det.values))
                                        mni = int(np.nanargmin(close_det.values))
                                        hv = float(close_det.iloc[int(h_i)]) if 0 <= int(h_i) < len(close_det) else float("nan")
                                        md.append(f"  - Detector window minClose: {mnv:.2f} at {d_local.index[mni]} | H_Close={hv:.2f}\n")
                                except Exception:
                                    pass
                                # max close between LS..H (exclusive)
                                if h_i > ls_i + 1:
                                    seg = d_local["Close"].iloc[ls_i+1:h_i]
                                    if len(seg):
                                        piv = []
                                        try:
                                            if patt == "HS_TOP":
                                                piv_all = _swing_highs_on_close(d_local, window=5, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
                                                piv = [int(x) for x in piv_all if int(ls_i) < int(x) < (int(h_i) - HS_MIN_SIDE_BARS)]
                                            elif patt == "IHS":
                                                piv_all = _swing_lows_on_close(d_local, window=5, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
                                                piv = [int(x) for x in piv_all if int(ls_i) < int(x) < (int(h_i) - HS_MIN_SIDE_BARS)]
                                        except Exception:
                                            piv = []
                                        if piv:
                                            vals = [(float(d_local["Close"].iloc[i]), int(i)) for i in piv]
                                            if patt == "HS_TOP":
                                                mx = max(v for v, _ in vals)
                                                cand = [i for v, i in vals if v == mx]
                                                j = int(max(cand)) if cand else int(piv[-1])
                                                md.append(f"  - Pre-head maxClose (swing pivot) between (LS,H): {float(mx):.2f} at {d_local.index[j]} | LS_Close={float(d_local['Close'].iloc[ls_i]):.2f}\n")
                                            else:
                                                mn = min(v for v, _ in vals)
                                                cand = [i for v, i in vals if v == mn]
                                                j = int(max(cand)) if cand else int(piv[-1])
                                                md.append(f"  - Pre-head minClose (swing pivot) between (LS,H): {float(mn):.2f} at {d_local.index[j]} | LS_Close={float(d_local['Close'].iloc[ls_i]):.2f}\n")
                                        else:
                                            j = int(seg.values.argmax()) + (ls_i + 1)
                                            md.append(f"  - Pre-head maxClose between (LS,H): {float(seg.max()):.2f} at {d_local.index[j]} | LS_Close={float(d_local['Close'].iloc[ls_i]):.2f}\n")
                                if rs_i > h_i + 1:
                                    seg2 = d_local["Close"].iloc[h_i+1:rs_i]
                                    if len(seg2):
                                        piv = []
                                        try:
                                            if patt == "HS_TOP":
                                                piv_all = _swing_highs_on_close(d_local, window=5, prominence_atr_mult=0.5, allow_tie_high_2dp=True)
                                                piv = [int(x) for x in piv_all if (int(h_i) + HS_MIN_SIDE_BARS) < int(x) < int(rs_i)]
                                            elif patt == "IHS":
                                                piv_all = _swing_lows_on_close(d_local, window=5, prominence_atr_mult=0.5, allow_tie_low_2dp=True)
                                                piv = [int(x) for x in piv_all if (int(h_i) + HS_MIN_SIDE_BARS) < int(x) < int(rs_i)]
                                        except Exception:
                                            piv = []
                                        if piv:
                                            vals = [(float(d_local["Close"].iloc[i]), int(i)) for i in piv]
                                            if patt == "HS_TOP":
                                                mx = max(v for v, _ in vals)
                                                cand = [i for v, i in vals if v == mx]
                                                j2 = int(min(cand)) if cand else int(piv[0])
                                                md.append(f"  - Post-head maxClose (swing pivot) between (H,RS): {float(mx):.2f} at {d_local.index[j2]} | RS_Close={float(d_local['Close'].iloc[rs_i]):.2f}\n")
                                            else:
                                                mn = min(v for v, _ in vals)
                                                cand = [i for v, i in vals if v == mn]
                                                j2 = int(min(cand)) if cand else int(piv[0])
                                                md.append(f"  - Post-head minClose (swing pivot) between (H,RS): {float(mn):.2f} at {d_local.index[j2]} | RS_Close={float(d_local['Close'].iloc[rs_i]):.2f}\n")
                                        else:
                                            j2 = int(seg2.values.argmax()) + (h_i + 1)
                                            md.append(f"  - Post-head maxClose between (H,RS): {float(seg2.max()):.2f} at {d_local.index[j2]} | RS_Close={float(d_local['Close'].iloc[rs_i]):.2f}\n")
                    except Exception:
                        pass
                except Exception:
                    pass
                # --- Validation gate history (3-gate daily log) ---
                try:
                    gate_hist = _focus_validation_gate_history(ft, df_ft, state=state, lookback_bars=25)
                    if gate_hist:
                        _gh_patt = gate_hist[0].get("_pattern", "")
                        _gh_dir = gate_hist[0].get("_direction", "")
                        _gh_exit = gate_hist[0].get("_exit_summary", "")
                        md.append(f"\n**Validation gate history** ({_gh_patt} / {_gh_dir})\n")
                        md.append(f"- Status: **{_gh_exit}**\n")
                        # Build HTML table — only show bars near the action (from first all3=True − 3 bars)
                        first_conf = None
                        for _gi, _gr in enumerate(gate_hist):
                            if _gr["all3"]:
                                first_conf = _gi
                                break
                        tbl_start = max(0, (first_conf or 0) - 3)
                        tbl_rows = gate_hist[tbl_start:]
                        md.append("<table style='border-collapse:collapse;font-size:13px;'>")
                        md.append("<tr style='background:#f0f0f0;'>"
                                  "<th style='padding:3px 8px;'>Date</th>"
                                  "<th style='padding:3px 8px;'>Close</th>"
                                  "<th style='padding:3px 8px;'>Trigger</th>"
                                  "<th style='padding:3px 8px;'>Dist(ATR)</th>"
                                  "<th style='padding:3px 8px;'>Price</th>"
                                  "<th style='padding:3px 8px;'>CLV</th>"
                                  "<th style='padding:3px 8px;'>CLV Gate</th>"
                                  "<th style='padding:3px 8px;'>Vol/Avg</th>"
                                  "<th style='padding:3px 8px;'>Vol Gate</th>"
                                  "<th style='padding:3px 8px;'>All 3</th>"
                                  "<th style='padding:3px 8px;'>Lifecycle</th>"
                                  "</tr>")
                        for _gr in tbl_rows:
                            _bg = ""
                            _lbl = _gr.get("lifecycle", "")
                            if "VALIDATED" in _lbl and "LOST" not in _lbl and "RECROSS" not in _lbl:
                                _bg = " style='background:#e6f4ea;'"
                            elif "CONFIRMED" in _lbl and "LOST" not in _lbl:
                                _bg = " style='background:#e8f0fe;'"
                            elif "LOST" in _lbl or "RECROSS" in _lbl or "FAILED" in _lbl:
                                _bg = " style='background:#fce8e6;'"
                            _p = "✓" if _gr["price_ok"] else "✗"
                            _c = "✓" if _gr["clv_ok"] else "✗"
                            _v = "✓" if _gr["vol_ok"] else "✗"
                            _a = "**✓**" if _gr["all3"] else "✗"
                            _pc = f"<span style='color:green;'>{_p}</span>" if _gr["price_ok"] else f"<span style='color:red;'>{_p}</span>"
                            _cc = f"<span style='color:green;'>{_c}</span>" if _gr["clv_ok"] else f"<span style='color:red;'>{_c}</span>"
                            _vc = f"<span style='color:green;'>{_v}</span>" if _gr["vol_ok"] else f"<span style='color:red;'>{_v}</span>"
                            _ac = f"<span style='color:green;font-weight:bold;'>✓</span>" if _gr["all3"] else f"<span style='color:red;'>✗</span>"
                            md.append(
                                f"<tr{_bg}>"
                                f"<td style='padding:2px 8px;'>{_gr['date']}</td>"
                                f"<td style='padding:2px 8px;text-align:right;'>{_gr['close']:.2f}</td>"
                                f"<td style='padding:2px 8px;text-align:right;'>{_gr['level']:.2f}</td>"
                                f"<td style='padding:2px 8px;text-align:right;'>{_gr['dist_atr']:+.2f}</td>"
                                f"<td style='padding:2px 8px;text-align:center;'>{_pc}</td>"
                                f"<td style='padding:2px 8px;text-align:right;'>{_gr['clv']:+.2f}</td>"
                                f"<td style='padding:2px 8px;text-align:center;'>{_cc}</td>"
                                f"<td style='padding:2px 8px;text-align:right;'>{_gr['vol_ratio']:.2f}×</td>"
                                f"<td style='padding:2px 8px;text-align:center;'>{_vc}</td>"
                                f"<td style='padding:2px 8px;text-align:center;'>{_ac}</td>"
                                f"<td style='padding:2px 8px;font-size:12px;'>{_lbl}</td>"
                                f"</tr>"
                            )
                        md.append("</table>\n")
                except Exception as _vge:
                    md.append(f"- Validation gate history error: {_vge}\n")
                sig = LevelSignal(
                    ticker=ft,
                    signal=f"FOCUS_{patt}_{direc}",
                    pattern=patt,
                    direction=direc,
                    level=lvl,
                    close=float(info.get("Close", float("nan"))),
                    atr=float(info.get("ATR", float("nan"))),
                    dist_atr=dist,
                    pct_today=float(info.get("Day%", float("nan"))),
                    meta=meta if isinstance(meta, dict) else None,
                )
            else:
                sig = LevelSignal(
                    ticker=ft,
                    signal="FOCUS_NO_CANDIDATE",
                    pattern="",
                    direction="",
                    level=float("nan"),
                    close=float(info.get("Close", float("nan"))),
                    atr=float(info.get("ATR", float("nan"))),
                    dist_atr=float("nan"),
                    pct_today=float(info.get("Day%", float("nan"))),
                    meta=None,
                )
            sig.chart_path = plot_signal_chart(ft, df_ft, sig, name_resolver=company_name_for_ticker)
            if getattr(sig, "chart_path", ""):
                md.append(f"<a href='{_cb_img(sig.chart_path)}' target='_blank'><img src='{_cb_img(sig.chart_path)}' width='980' style='max-width:980px;height:auto;'></a>\n")
            md.append("\n")
        except Exception as e:
            md.append(f"**{ft}** — focus analysis failed: `{e}`\n\n")
    md.append(build_watchlist_pulse_section_md(
        df_early_new=df_early_new,
        df_early_old=df_early_old,
        df_conf_new=df_conf_new,
        df_conf_old=df_conf_old,
        df_val_new=df_val_new,
        df_val_old=df_val_old,
        watchlist_groups=WATCHLIST_GROUPS,
        ticker_labels=TICKER_LABELS,
    ))
    md.append("### 4B) Early callouts (~90% complete)\n")
    md.append("_Close enough to pre-plan. “Close enough” = within 0.5 ATR of the trigger (neckline/boundary). No SOFT tier — anything not CONFIRMED stays in EARLY._\n")
    md.append("**NEW (today):**\n")
    df_early_new_tbl = df_early_new.copy()
    if "Level" in df_early_new_tbl.columns and "Threshold" not in df_early_new_tbl.columns:
        df_early_new_tbl["Threshold"] = df_early_new_tbl["Level"]
    md.append(html_table_from_df(df_early_new_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "Chart"], max_rows=60))
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
                level_v = float(rr.get("Threshold"))
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
                md.append(f'<img src="{_cb_img(chart_p)}" width="720" style="max-width:100%;height:auto;">')
            md.append("")
    md.append("\n**ONGOING:**\n")
    df_early_old_tbl = df_early_old.copy()
    if "Level" in df_early_old_tbl.columns and "Threshold" not in df_early_old_tbl.columns:
        df_early_old_tbl["Threshold"] = df_early_old_tbl["Level"]
    md.append(html_table_from_df(df_early_old_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "Chart"], max_rows=120))
    if ended_by_stage.get("EARLY"):
        md.append("\n**Ended today (EARLY):**\n")
        for x in ended_by_stage["EARLY"][:120]:
            md.append(f"- {x}")
        md.append("")
    md.append("")
    md.append("### 4C) Confirmed breakouts / breakdowns (watchlist + MSCI World + EM)\n")
    md.append("_Includes **CONFIRMED** only: **Day 0 (NEW)** requires **all 3** gates (Price+CLV+Volume). **Day 1 (ONGOING)** requires **Price+Volume** (CLV optional). All tickers use S&P 500 11-sector labels (Sector)._ \n")
    md.append("**NEW (today):**\n")
    df_conf_new_tbl = df_conf_new.copy()
    if "Level" in df_conf_new_tbl.columns and "Threshold" not in df_conf_new_tbl.columns:
        df_conf_new_tbl["Threshold"] = df_conf_new_tbl["Level"]
    if not df_conf_new_tbl.empty and "Sector" in df_conf_new_tbl.columns:
        df_conf_new_tbl = df_conf_new_tbl.sort_values(["Sector", "Signal", "Dist(ATR)"], na_position="last")
    md.append(html_table_from_df(df_conf_new_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "HVN Runway%", "Chart"], max_rows=120))
    md.append("\n**ONGOING:**\n")
    df_conf_old_tbl = df_conf_old.copy()
    if "Level" in df_conf_old_tbl.columns and "Threshold" not in df_conf_old_tbl.columns:
        df_conf_old_tbl["Threshold"] = df_conf_old_tbl["Level"]
    if not df_conf_old_tbl.empty and "Sector" in df_conf_old_tbl.columns:
        df_conf_old_tbl = df_conf_old_tbl.sort_values(["Sector", "Signal", "Dist(ATR)"], na_position="last")
    md.append(html_table_from_df(df_conf_old_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "HVN Runway%", "Chart"], max_rows=240))
    if ended_by_stage.get("CONFIRMED"):
        md.append("\n**Ended today (CONFIRMED):**\n")
        for x in ended_by_stage["CONFIRMED"][:120]:
            md.append(f"- {x}")
        md.append("")
    md.append("")
    # 5) Catalysts
    md.append("")
    md.append("### 4D) Validated breakouts / breakdowns (3-session anti-whipsaw)\n")
    md.append("_Includes **VALIDATED** only: 3-day lifecycle where day0 (confirmation) = **all 3** gates, day1 = **price+volume** (CLV optional), day2 = **all 3** gates again. (Then VALIDATED can remain ongoing while price stays beyond the trigger.) **HVN Runway%** = distance from current price to the nearest significant opposing Volume-Profile HVN zone (daily OHLCV approximation), expressed as % in the signal direction._\n")
    md.append("**NEW (today):**\n")
    df_val_new_tbl = df_val_new.copy()
    if "Level" in df_val_new_tbl.columns and "Threshold" not in df_val_new_tbl.columns:
        df_val_new_tbl["Threshold"] = df_val_new_tbl["Level"]
    if not df_val_new_tbl.empty and "Sector" in df_val_new_tbl.columns:
        df_val_new_tbl = df_val_new_tbl.sort_values(["Sector", "Signal", "HVN Runway%", "Dist(ATR)"], ascending=[True, True, False, True], na_position="last")
    md.append(html_table_from_df(df_val_new_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "HVN Runway%", "Chart"], max_rows=80))
    md.append("\n**ONGOING:**\n")
    df_val_old_tbl = df_val_old.copy()
    if "Level" in df_val_old_tbl.columns and "Threshold" not in df_val_old_tbl.columns:
        df_val_old_tbl["Threshold"] = df_val_old_tbl["Level"]
    if not df_val_old_tbl.empty and "Sector" in df_val_old_tbl.columns:
        df_val_old_tbl = df_val_old_tbl.sort_values(["Sector", "Signal", "HVN Runway%", "Dist(ATR)"], ascending=[True, True, False, True], na_position="last")
    md.append(html_table_from_df(df_val_old_tbl, cols=["Name of Company", "Ticker", "Country", "Sector", "Signal", "Close", "Day%", "Threshold", "CLV", "ATR(14)", "Dist(ATR)", "Vol/AvgVol(20)", "HVN Runway%", "Chart"], max_rows=160))
    if ended_by_stage.get("VALIDATED"):
        md.append("\n**Ended today (VALIDATED):**\n")
        for x in ended_by_stage["VALIDATED"][:120]:
            md.append(f"- {x}")
        md.append("")
    md.append("")
    md.append("## 5) Needle-moving catalysts (RSS digest)\n")
    md.append("_Linked digest for drill-down._\n")
    md.append(format_rss_digest(rss_items, max_items=10))
    md.append("")
    # Section 6: Watchlist performance
    md.append(build_watchlist_performance_section_md(
        ohlcv,
        sector_resolver,
        name_resolver=company_name_for_ticker,
        country_resolver=country_for_ticker,
    ))

    # GC Ignition Scoring Pass — scores Star 2/3 for all Star 1 tickers,
    # writes ignition_signals to gc_state.json so Section 7 can render.
    try:
        gc_state_path = DOCS_DIR / "gc_state.json"
        if gc_state_path.exists():
            run_gc_ignition_scoring(
                all_signals=all_signals,
                ohlcv=ohlcv,
                gc_state_path=gc_state_path,
                name_resolver=company_name_for_ticker,
                sector_resolver=sector_resolver,
            )
    except Exception as _gc_err:
        print(f"[gc-scoring] error: {_gc_err}")

    # Section 7: Growth Compounders — Three-Star Signals
    try:
        gc_state_path = DOCS_DIR / "gc_state.json"
        if gc_state_path.exists():
            s7 = build_gc_three_star_section_md(gc_state_path)
            if s7:
                md.append(s7)
            else:
                # Always show section header so it's visible in the report
                md.append("\n## 7) Growth Compounders — Three-Star Signals\n")
                md.append("_No three-star signals today. Three-star requires: (1) technical ignition, ")
                md.append("(2) dual EPS+Revenue beat ≥2Q or massive catalyst, ")
                md.append("(3) revenue YoY ≥20% with a durable moat (AI-confirmed)._\n")
    except Exception as _s7_err:
        print(f"[section7] error: {_s7_err}")

    md_text = "\n".join(md).strip() + "\n"
    md_text = _dedupe_macro_cards(md_text)
    # Web page keeps macro charts
    write_text(INDEX_PATH, md_text)
    # Email/report version — keep macro charts inline (stripping them caused email clients to
    # render them as bottom-of-email attachments instead of inline in section 1).
    write_text(REPORT_PATH, md_text)
    write_text(EMAIL_REPORT_PATH, md_text)
    print(f"Wrote: {REPORT_PATH}")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Universe(base={len(base_universe)}, tech_scan={len(tech_scan_universe)})  Signals: early={len(early_sorted)} confirmed={len(confirmed_sorted)} validated={len(validated_sorted)}")
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
