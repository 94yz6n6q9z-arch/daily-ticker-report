#!/usr/bin/env python3
"""
export_universe.py  —  Universe review export  v2
==================================================
Produces universe_review.csv: every ticker in the MSCI World + EM universe
with country, exchange, market cap, current gc_state status, and data health.

Fully standalone — no universe.py / gc_engine.py imports required.

Usage:
    python export_universe.py [--state docs/gc_state.json] [--out universe_review.csv]

Changes in v2:
  - Fixed 14 wrong exchange→country mappings (.AT Greece, .VI Austria, .CA Egypt,
    .BD Hungary, .PR Czech Republic, .R Thailand NVDRs, .CL Colombia, .SI Singapore,
    .TA Israel, .B Colombia B-share, .SN Chile, .LS Portugal, .IR Ireland, .RE Reunion)
  - _last_known_mcap fallback: uses Fix-A field when info.market_cap is absent
  - Ghost-risk summary now flags no-data tickers from outage that could tip to inactive
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR   = BASE_DIR / "docs"

DEFAULT_STATE = DOCS_DIR / "gc_state.json"
DEFAULT_OUT   = BASE_DIR / "universe_review.csv"

MIN_MCAP_US_EU = 2_000_000_000
MIN_MCAP_OTHER = 5_000_000_000

EU_SUFFIXES = {
    "L", "DE", "PA", "AS", "MI", "MC", "ST", "OL", "HE", "CO",
    "LS", "BR", "AT", "IR", "SW", "WA", "VI",
}

# ── Correct exchange→country map ─────────────────────────────────────────────
# KEY CORRECTIONS vs v1:
#   .AT  = Athens Exchange (Greece)  — NOT Vienna (that is .VI)
#   .VI  = Vienna Stock Exchange (Austria)
#   .CA  = Egyptian Exchange Cairo   — NOT Chile
#   .BD  = Budapest Stock Exchange (Hungary)  — NOT Bangladesh
#   .PR  = Prague Stock Exchange (Czech Republic)
#   .R   = Thailand NVDR share class — NOT Russia
#   .CL  = Colombia (Bolsa de Valores de Colombia) — NOT Chile
#   .SN  = Santiago Stock Exchange (Chile)
#   .SI  = Singapore Exchange (SGX)
#   .TA  = Tel Aviv Stock Exchange (Israel)
#   .B   = Colombia B-share (or Santiago B-share — confirm with data)
EXCHANGE_COUNTRY = {
    # Americas
    "US": "United States",
    "TO": "Canada",
    "SA": "Brazil",
    "MX": "Mexico",
    "SN": "Chile (Santiago)",       # .SN = Bolsa de Santiago
    "CL": "Colombia",               # .CL = Bolsa de Valores de Colombia (NOT Chile)
    "CA": "Egypt",                  # .CA = Egyptian Exchange (Cairo) — NOT Canada/Chile
    "B":  "Colombia (B share)",     # ANDINA.B = Embotelladora Andina Colombia B share

    # Europe
    "L":  "United Kingdom",
    "DE": "Germany",
    "PA": "France",
    "AS": "Netherlands",
    "MI": "Italy",
    "MC": "Spain",
    "ST": "Sweden",
    "OL": "Norway",
    "HE": "Finland",
    "CO": "Denmark",
    "LS": "Portugal",
    "BR": "Belgium",
    "IR": "Ireland",
    "SW": "Switzerland",
    "WA": "Poland",
    "VI": "Austria",                # .VI = Vienna Stock Exchange (Wiener Börse)
    "AT": "Greece",                 # .AT = Athens Exchange (ATHEX) — NOT Austria
    "PR": "Czech Republic",         # .PR = Prague Stock Exchange
    "BD": "Hungary",                # .BD = Budapest Stock Exchange — NOT Bangladesh
    "GR": "Greece (alt)",           # fallback if any .GR appears
    "WA": "Poland",

    # Middle East / Africa
    "SR": "Saudi Arabia",
    "QA": "Qatar",
    "KW": "Kuwait",
    "AD": "UAE Abu Dhabi (dead)",
    "DU": "UAE Dubai (dead)",
    "JO": "South Africa",
    "EG": "Egypt (alt)",
    "IL": "Israel (alt)",
    "TA": "Israel",                  # .TA = Tel Aviv Stock Exchange

    # Asia-Pacific
    "T":  "Japan",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "SS": "China (Shanghai)",
    "SZ": "China (Shenzhen)",
    "KS": "South Korea",
    "NS": "India (NSE)",
    "BO": "India (BSE)",
    "SI": "Singapore",              # .SI = SGX Singapore
    "AX": "Australia",
    "NZ": "New Zealand",
    "KL": "Malaysia (dead)",        # Bursa Malaysia — dead exchange
    "PS": "Philippines (dead)",     # PSEi — dead exchange
    "JK": "Indonesia",
    "BK": "Thailand",
    "R":  "Thailand (NVDR)",        # .R = Thai NVDR share class — NOT Russia
    "SG": "Singapore (alt)",

    # Other / edge
    "IS": "Turkey",
    "RE": "Reunion",
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
    "market_cap_b", "mcap_source", "mcap_floor_b", "passes_mcap",
    "status", "no_data_runs", "ghost_risk",
    "has_eps", "has_revenue", "data_gap",
    "inactive_reason", "inactive_since", "fetched_at", "error",
]

# Known data quality issues — flagged in output
KNOWN_BAD_CSVS = {
    "korea": "⚠ Contains Indian .NS tickers, not Korean .KS — needs regeneration",
    "taiwan": "⚠ Contains bare US tickers without .TW suffix — needs regeneration",
}


def _exch(ticker):
    return ticker.rsplit(".", 1)[-1] if "." in ticker else "US"


def _country(ticker):
    ex = _exch(ticker)
    return EXCHANGE_COUNTRY.get(ex, f"Unknown ({ex})")


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
            t = (row.get("Ticker") or row.get("ticker") or "").strip()
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

    cache = {}
    if state_path.exists():
        print(f"[export] Loading {state_path} …")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cache = state.get("earnings_cache", {})
        print(f"[export] gc_state cache entries: {len(cache):,}")
    else:
        print(f"[export] WARNING: {state_path} not found")

    csv_sources = defaultdict(set)
    csv_counts  = {}

    for name, fname in CSV_DEFS:
        path = CONFIG_DIR / fname
        tickers = _read_csv_tickers(path)
        csv_counts[name] = len(tickers)
        for t in tickers:
            csv_sources[t].add(name)
        if name in KNOWN_BAD_CSVS and tickers:
            print(f"[export] WARNING {name} CSV: {KNOWN_BAD_CSVS[name]}")

    for ticker in cache:
        if ticker not in csv_sources:
            csv_sources[ticker].add("state_only")

    all_tickers = sorted(csv_sources.keys())

    print(f"\n[export] Raw CSV counts (before dedup):")
    for name, _ in CSV_DEFS:
        n = csv_counts.get(name, 0)
        note = "  ← KNOWN BAD" if name in KNOWN_BAD_CSVS else ""
        print(f"  {name:8s}: {n:5,}{note}")
    print(f"  Raw sum:      {sum(csv_counts.values()):,}")
    print(f"  After dedup:  {len(all_tickers):,} unique tickers")
    overlap = sum(1 for s in csv_sources.values() if len(s) > 1)
    print(f"  Multi-CSV:    {overlap:,} tickers appear in 2+ CSVs")
    print(f"  Explanation:  Taiwan/Korea/HK/China tickers appear in both their")
    print(f"                country CSV and the EM CSV. Dedup collapses to one.")

    risk_1 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 1)
    risk_2 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 2)
    print(f"\n[export] Pre-poisoned _no_data_runs from outage runs:")
    print(f"  _no_data_runs=1 (MED risk):  {risk_1:,}  — 2 more misses → false-inactive")
    print(f"  _no_data_runs=2 (HIGH risk): {risk_2:,}  — 1 more miss  → false-inactive")

    rows = []
    for ticker in all_tickers:
        entry   = cache.get(ticker, {})
        info    = entry.get("info") or {}
        sources = csv_sources[ticker]

        floor_raw   = _mcap_floor(ticker)
        floor_b_str = f"{floor_raw / 1e9:.0f}"

        # Fix A fallback: prefer info.market_cap, then _last_known_mcap
        mcap_raw    = info.get("market_cap") or entry.get("_last_known_mcap")
        mcap_source = ""
        if info.get("market_cap"):
            mcap_source = "yf"
        elif entry.get("_last_known_mcap"):
            mcap_source = entry.get("_mcap_source", "cached")
        mcap_b_str  = _fmt_b(mcap_raw)

        try:
            mcap_float = float(mcap_raw) if mcap_raw else 0.0
        except (TypeError, ValueError):
            mcap_float = 0.0

        passes_mcap = "?" if mcap_float <= 0 else ("Y" if mcap_float >= floor_raw else "N")

        # Status
        if entry.get("inactive"):
            ir    = entry.get("inactive_reason", "")
            since = (entry.get("inactive_since") or "")[:10]
            if any(x in ir for x in ["no_price_no_financials", "known_dead", "dead_market"]):
                status = "inactive_dead"
            elif ir == "persistent_no_data_3_runs" and since >= "2026-03-11":
                status = "inactive_degraded"
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

        ndr        = entry.get("_no_data_runs", 0)
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
            "mcap_source":     mcap_source,
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

    sc = Counter(r["status"] for r in rows)
    print(f"\n[export] Status breakdown:")
    for s, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {n:5,}")

    mc_yes = sum(1 for r in rows if r["passes_mcap"] == "Y")
    mc_no  = sum(1 for r in rows if r["passes_mcap"] == "N")
    mc_unk = sum(1 for r in rows if r["passes_mcap"] == "?")
    print(f"\n[export] Market cap filter:")
    print(f"  Passes floor ($B threshold): {mc_yes:,}")
    print(f"  Below floor:                 {mc_no:,}")
    print(f"  No mcap data:                {mc_unk:,}  ← run again after tonight\'s clean fetch")

    print(f"\n[export] Active tickers by country (top 25):")
    ca = Counter(r["country"] for r in rows if r["status"] == "active")
    for country, n in ca.most_common(25):
        print(f"  {country:35s}: {n:4,}")

    print(f"\n[export] Known CSV issues to fix:")
    for name, msg in KNOWN_BAD_CSVS.items():
        print(f"  {name:8s}: {msg}")

    print(f"\n[export] Done.")


if __name__ == "__main__":
    main()
