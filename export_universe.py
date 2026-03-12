#!/usr/bin/env python3
"""
export_universe.py  —  Universe review export
==============================================
Produces universe_review.csv: every ticker in the MSCI World + EM universe
with country, exchange, market cap, current gc_state status, and data health.

Fully standalone — no universe.py / gc_engine.py imports required.

Usage:
    python export_universe.py [--state docs/gc_state.json] [--out universe_review.csv]

Output columns:
    ticker            Yahoo Finance symbol
    country           Human-readable country name
    exchange          Exchange suffix (US, L, TW, T, HK, KS, SS, SZ, …)
    source_csvs       Which config CSVs contain this ticker (comma-separated)
    in_em_csv         Y if also in msci_em_classification.csv (overlap flag)
    market_cap_b      Market cap in $B from last cached fetch (blank = never fetched)
    mcap_floor_b      Minimum investable mcap for this exchange ($B)
    passes_mcap       Y / N / ? (? = no market cap in cache yet)
    status            active | below_min_mcap | inactive_dead |
                      inactive_ghost | inactive_degraded | no_cache
    no_data_runs      _no_data_runs counter (1/2 = pre-poisoned from outage)
    ghost_risk        HIGH if no_data_runs>=2, MED if ==1, blank otherwise
    has_eps           Y if gc_state has any EPS reported history
    has_revenue       Y if gc_state has >=4 quarters of revenue
    data_gap          Y if completely empty (no eps, no rev, no info)
    inactive_reason   If inactive: why
    inactive_since    If inactive: date (YYYY-MM-DD)
    fetched_at        Last fetch attempt date (YYYY-MM-DD)
    error             Last error string if any (truncated to 80 chars)
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR   = BASE_DIR / "docs"

DEFAULT_STATE = DOCS_DIR / "gc_state.json"
DEFAULT_OUT   = BASE_DIR / "universe_review.csv"

MIN_MCAP_US_EU = 2_000_000_000
MIN_MCAP_OTHER = 5_000_000_000

EU_SUFFIXES = {
    "L", "DE", "PA", "AS", "MI", "MC", "ST", "OL", "HE", "CO",
    "LS", "BR", "AT", "IR", "SW", "WA",
}

EXCHANGE_COUNTRY = {
    "US": "United States",   "L":  "United Kingdom",  "T":  "Japan",
    "TW": "Taiwan",          "HK": "Hong Kong",        "SS": "China (Shanghai)",
    "SZ": "China (Shenzhen)","KS": "South Korea",      "SR": "Saudi Arabia",
    "NS": "India (NSE)",     "BO": "India (BSE)",      "SA": "Brazil",
    "TO": "Canada",          "AX": "Australia",        "ST": "Sweden",
    "SW": "Switzerland",     "DE": "Germany",          "PA": "France",
    "AS": "Netherlands",     "MI": "Italy",            "MC": "Spain",
    "HE": "Finland",         "CO": "Denmark",          "OL": "Norway",
    "WA": "Poland",          "AT": "Austria",          "IR": "Ireland",
    "LS": "Portugal",        "BR": "Belgium",          "JK": "Indonesia",
    "BK": "Thailand",        "KL": "Malaysia (dead)",  "IS": "Turkey",
    "JO": "South Africa",    "QA": "Qatar",            "KW": "Kuwait",
    "CA": "Chile",           "MX": "Mexico",           "NZ": "New Zealand",
    "SG": "Singapore",       "AD": "UAE Abu Dhabi (dead)", "DU": "UAE Dubai (dead)",
    "PS": "Philippines (dead)", "GR": "Greece",        "EG": "Egypt",
    "BD": "Bangladesh",      "IL": "Israel",           "RE": "Reunion",
    "R":  "Russia (dead)",   "B":  "Bulgaria",
}

CSV_DEFS = [
    ("world",  "msci_world_classification.csv"),
    ("em",     "msci_em_classification.csv"),
    ("japan",  "msci_japan_classification.csv"),
    ("taiwan", "msci_taiwan_classification.csv"),
    ("china",  "msci_china_classification.csv"),
    ("hk",     "msci_hk_classification.csv"),
    ("saudi",  "msci_saudi_classification.csv"),
    ("korea",  "msci_korea_classification.csv"),
    ("nzl",    "msci_nzl_classification.csv"),
]

FIELDNAMES = [
    "ticker", "country", "exchange", "source_csvs", "in_em_csv",
    "market_cap_b", "mcap_floor_b", "passes_mcap",
    "status", "no_data_runs", "ghost_risk",
    "has_eps", "has_revenue", "data_gap",
    "inactive_reason", "inactive_since", "fetched_at", "error",
]


def _exch(ticker):
    return ticker.rsplit(".", 1)[-1] if "." in ticker else "US"

def _country(ticker):
    return EXCHANGE_COUNTRY.get(_exch(ticker), _exch(ticker))

def _mcap_floor(ticker):
    ex = _exch(ticker)
    return MIN_MCAP_US_EU if (ex == "US" or ex in EU_SUFFIXES) else MIN_MCAP_OTHER

def _fmt_b(v):
    try:
        f = float(v)
        if f > 0:
            return f"{f / 1e9:.2f}"
    except (TypeError, ValueError):
        pass
    return ""

def _read_csv_tickers(path):
    if not path.exists():
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or row.get("Ticker") or "").strip()
            if t:
                result.append(t)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--out",   default=str(DEFAULT_OUT))
    args = ap.parse_args()

    state_path = Path(args.state)
    out_path   = Path(args.out)

    # Load gc_state
    cache = {}
    if state_path.exists():
        print(f"[export] Loading {state_path} …")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cache = state.get("earnings_cache", {})
        print(f"[export] gc_state cache entries: {len(cache):,}")
    else:
        print(f"[export] WARNING: {state_path} not found")

    # Load CSVs
    csv_sources = defaultdict(set)   # ticker → set of csv names
    csv_counts  = {}

    for name, fname in CSV_DEFS:
        path = CONFIG_DIR / fname
        tickers = _read_csv_tickers(path)
        csv_counts[name] = len(tickers)
        for t in tickers:
            csv_sources[t].add(name)

    for ticker in cache:
        if ticker not in csv_sources:
            csv_sources[ticker].add("state_only")

    all_tickers = sorted(csv_sources.keys())

    # Print overlap explanation
    print(f"\n[export] Raw CSV counts (before dedup):")
    for name, _ in CSV_DEFS:
        n = csv_counts.get(name, 0)
        print(f"  {name:8s}: {n:5,}")
    print(f"  Raw sum:  {sum(csv_counts.values()):,}")
    print(f"  After dedup: {len(all_tickers):,} unique tickers")
    overlap = sum(1 for s in csv_sources.values() if len(s) > 1)
    print(f"  Overlap (in 2+ CSVs): {overlap:,}")
    print(f"  Explanation: Taiwan/Korea/HK/China tickers appear in both")
    print(f"  their country CSV and the EM CSV. Dedup collapses them to one.")

    # Ghost risk
    risk_1 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 1)
    risk_2 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 2)
    print(f"\n[export] Pre-poisoned _no_data_runs from outage (not cleared by repair):")
    print(f"  _no_data_runs=1 (MED risk): {risk_1:,}  — need 2 more misses to false-inactive")
    print(f"  _no_data_runs=2 (HIGH risk): {risk_2:,}  — need 1 more miss to false-inactive")
    print(f"  These reset automatically when ticker returns real data.")

    # Build rows
    rows = []
    for ticker in all_tickers:
        entry   = cache.get(ticker, {})
        info    = entry.get("info") or {}
        sources = csv_sources[ticker]

        floor_raw   = _mcap_floor(ticker)
        floor_b_str = f"{floor_raw / 1e9:.0f}"
        mcap_raw    = info.get("market_cap")
        mcap_b_str  = _fmt_b(mcap_raw)

        try:
            mcap_float = float(mcap_raw) if mcap_raw else 0.0
        except (TypeError, ValueError):
            mcap_float = 0.0

        passes_mcap = "?" if mcap_float <= 0 else ("Y" if mcap_float >= floor_raw else "N")

        # Status classification
        if entry.get("inactive"):
            ir    = entry.get("inactive_reason", "")
            since = (entry.get("inactive_since") or "")[:10]
            if any(x in ir for x in ["no_price_no_financials", "known_dead", "dead_market"]):
                status = "inactive_dead"
            elif ir == "persistent_no_data_3_runs" and since >= "2026-03-11":
                status = "inactive_degraded"   # should have been repaired — flag if any remain
            elif ir == "persistent_no_data_3_runs":
                status = "inactive_ghost"
            else:
                status = "inactive_other"
        elif entry.get("below_min_mcap"):
            status = "below_min_mcap"
        elif not entry:
            status = "no_cache"
        else:
            status = "active"

        ndr = entry.get("_no_data_runs", 0)
        ghost_risk = "HIGH" if ndr >= 2 else ("MED" if ndr == 1 else "")

        has_eps = any(
            e.get("eps_reported") is not None
            for e in entry.get("earnings_dates", [])
        )
        has_rev  = len(entry.get("quarterly_revenue", [])) >= 4
        has_info = bool(info.get("revenue_growth") or info.get("market_cap"))
        data_gap = not (has_eps or has_rev or has_info)

        rows.append({
            "ticker":          ticker,
            "country":         _country(ticker),
            "exchange":        _exch(ticker),
            "source_csvs":     ",".join(sorted(sources)),
            "in_em_csv":       "Y" if "em" in sources else "",
            "market_cap_b":    mcap_b_str,
            "mcap_floor_b":    floor_b_str,
            "passes_mcap":     passes_mcap,
            "status":          status,
            "no_data_runs":    str(ndr) if ndr else "",
            "ghost_risk":      ghost_risk,
            "has_eps":         "Y" if has_eps else "",
            "has_revenue":     "Y" if has_rev else "",
            "data_gap":        "Y" if data_gap else "",
            "inactive_reason": entry.get("inactive_reason", ""),
            "inactive_since":  (entry.get("inactive_since") or "")[:10],
            "fetched_at":      (entry.get("fetched_at") or "")[:10],
            "error":           str(entry.get("error", ""))[:80],
        })

    rows.sort(key=lambda r: (r["country"], r["ticker"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[export] Written {len(rows):,} rows → {out_path}")

    # Summary
    sc = Counter(r["status"] for r in rows)
    print(f"\n[export] Status breakdown:")
    for s, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {n:5,}")

    mc_yes = sum(1 for r in rows if r["passes_mcap"] == "Y")
    mc_no  = sum(1 for r in rows if r["passes_mcap"] == "N")
    mc_unk = sum(1 for r in rows if r["passes_mcap"] == "?")
    print(f"\n[export] Market cap filter (cached mcap):")
    print(f"  Passes floor:     {mc_yes:,}")
    print(f"  Below floor:      {mc_no:,}  ← review these in Excel: filter passes_mcap=N")
    print(f"  No cached mcap:   {mc_unk:,}  ← will be fetched and evaluated next run")

    print(f"\n[export] Active tickers by country (top 20):")
    ca = Counter(r["country"] for r in rows if r["status"] == "active")
    for country, n in ca.most_common(20):
        print(f"  {country:32s}: {n:4,}")

    print(f"\n[export] Done. Open universe_review.csv in Excel to decide on mcap floors.")


if __name__ == "__main__":
    main()
