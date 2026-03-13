#!/usr/bin/env python3
"""
export_universe.py  —  Universe review export  v4
==================================================
Produces universe_review.csv: every ticker in the MSCI World + EM universe
with country, exchange, market cap, current gc_state status, and data health.

Usage:
    python export_universe.py [--state docs/gc_state.json] [--out universe_review.csv]

Changes in v4:
  - ROOT FIX: Import universe._normalize_ticker and apply to all CSV-sourced tickers
    before building csv_sources. Previously AKBNK.E.IS (raw XLS) never matched
    AKBNK.IS (cache key) — 100+ tickers falsely showed as no_cache. Now they match.
  - EXCHANGE_COUNTRY: added AE (UAE unified), KQ (KOSDAQ), TWO (Taiwan Gretai)
  - mcap: use _mcap_usd (FX-converted by gc_engine) for market_cap_usd_b;
    mcap_local_b + mcap_currency kept as separate columns
  - Unified $2B USD mcap floor (was split $2B US/EU + $5B EM)
  - passes_mcap=? only when _mcap_usd absent; N only when confirmed below floor
  - Full v3 column schema: q1..q4 historical + fwd_* forward estimates,
    has_eps_history, has_rev_history, has_fwd_eps, has_fwd_rev, company

Changes in v3 (carried forward):
  - q1..q4 historical quarter columns (date, eps/rev reported+estimate+beat+source)
  - fwd_* forward estimate columns
  - has_eps_history / has_rev_history (actuals only)
  - has_fwd_eps / has_fwd_rev (forward consensus)

Changes in v2 (carried forward):
  - Fixed 14 wrong exchange→country mappings
  - _last_known_mcap fallback; ghost-risk summary
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import normalization from universe.py (same directory)
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from universe import _normalize_ticker, TICKER_OVERRIDES
    _HAS_UNIVERSE = True
except ImportError:
    _HAS_UNIVERSE = False
    print("[export] WARNING: universe.py not found — ticker normalization disabled, no_cache may be inflated")
    def _normalize_ticker(t): return t
    TICKER_OVERRIDES = {}

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR   = BASE_DIR / "docs"

DEFAULT_STATE = DOCS_DIR / "gc_state.json"
DEFAULT_OUT   = BASE_DIR / "universe_review.csv"

# Unified $1B USD floor (gc_engine v0.8.0 — FX conversion handles currency)
MIN_MCAP_USD = 1_000_000_000

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
    "SN": "Chile (Santiago)",
    "CL": "Colombia",
    "CA": "Egypt",                  # .CA = Egyptian Exchange (Cairo) — NOT Canada/Chile
    "B":  "Colombia (B share)",

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
    "VI": "Austria",
    "AT": "Greece",                 # .AT = Athens Exchange (ATHEX) — NOT Austria
    "PR": "Czech Republic",
    "BD": "Hungary",
    "GR": "Greece (alt)",

    # Middle East / Africa
    "SR": "Saudi Arabia",
    "QA": "Qatar",
    "KW": "Kuwait",
    "AE": "UAE",                    # .AE = unified after universe.py remaps .AD/.DU
    "AD": "UAE (Abu Dhabi)",        # legacy — universe.py remaps to .AE
    "DU": "UAE (Dubai)",            # legacy — universe.py remaps to .AE
    "JO": "South Africa",
    "EG": "Egypt (alt)",
    "IL": "Israel (alt)",
    "TA": "Israel",

    # Asia-Pacific
    "T":   "Japan",
    "TW":  "Taiwan",
    "TWO": "Taiwan (Gretai OTC)",   # .TWO = Taipei Exchange / Gretai Securities Market
    "HK":  "Hong Kong",
    "SS":  "China (Shanghai)",
    "SZ":  "China (Shenzhen)",
    "KS":  "South Korea",
    "KQ":  "South Korea (KOSDAQ)",  # .KQ = Korea KOSDAQ
    "NS":  "India (NSE)",
    "BO":  "India (BSE)",
    "SI":  "Singapore",
    "AX":  "Australia",
    "NZ":  "New Zealand",
    "KL":  "Malaysia (dead)",
    "PS":  "Philippines (dead)",
    "JK":  "Indonesia",
    "BK":  "Thailand",
    "R":   "Thailand (NVDR)",
    "SG":  "Singapore (alt)",

    # Other
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
    "ticker", "company", "country", "exchange", "source_csvs", "in_em_csv",
    "market_cap_usd_b", "mcap_local_b", "mcap_currency", "mcap_source", "passes_mcap",
    "status", "no_data_runs", "ghost_risk",
    "has_eps_history", "has_rev_history", "has_fwd_eps", "has_fwd_rev", "data_gap",
    # Most recent historical quarter (q1 = newest)
    "q1_date", "q1_eps_reported", "q1_eps_estimate", "q1_eps_beat", "q1_eps_source",
    "q1_rev_reported", "q1_rev_estimate", "q1_rev_beat", "q1_rev_source",
    # Second most recent quarter
    "q2_date", "q2_eps_reported", "q2_eps_estimate", "q2_eps_beat", "q2_eps_source",
    "q2_rev_reported", "q2_rev_estimate", "q2_rev_beat", "q2_rev_source",
    # Third
    "q3_date", "q3_eps_reported", "q3_eps_estimate", "q3_eps_beat", "q3_eps_source",
    "q3_rev_reported", "q3_rev_estimate", "q3_rev_beat", "q3_rev_source",
    # Fourth
    "q4_date", "q4_eps_reported", "q4_eps_estimate", "q4_eps_beat", "q4_eps_source",
    "q4_rev_reported", "q4_rev_estimate", "q4_rev_beat", "q4_rev_source",
    # Nearest upcoming quarter (forward — no actuals/beat yet)
    "fwd_date", "fwd_eps_estimate", "fwd_eps_source", "fwd_rev_estimate", "fwd_rev_source",
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
    # Unified $2B USD floor — FX conversion handled by gc_engine
    return MIN_MCAP_USD


def _fmt_b(v):
    try:
        f = float(v)
        if f > 0:
            return f"{f / 1e9:.2f}"
    except (TypeError, ValueError):
        pass
    return ""


def _pct(v):
    """Format a float as percentage string e.g. 0.053 → '5.3%'"""
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _beat_str(reported, estimate):
    """Return 'beat'/'miss'/'' from two numeric strings."""
    try:
        r, e = float(reported), float(estimate)
        if abs(e) < 1e-9:
            return ""
        return "beat" if r >= e else "miss"
    except (TypeError, ValueError):
        return ""


def _quarter_row(q: dict, rev_row: dict | None) -> dict:
    """Extract one quarter's worth of columns from an earnings_dates entry + optional rev row."""
    import datetime as _dt
    today = _dt.date.today().isoformat()
    date = (q.get("date") or "")[:10]
    if not date or date > today:
        return {}  # forward quarter — don't include as historical

    eps_rep  = q.get("eps_reported")
    eps_est  = q.get("eps_estimate")
    eps_src  = q.get("_eps_est_source") or ("yfinance" if eps_rep is not None else "")
    eps_beat = _beat_str(eps_rep, eps_est) if eps_est is not None else ""

    rev_rep  = q.get("revenue_reported")
    rev_est  = q.get("revenue_estimate")
    rev_src  = q.get("_rev_source") or ""
    # Fallback: pull from matched quarterly_revenue row
    if rev_rep is None and rev_row:
        rev_rep = rev_row.get("revenue")
        if rev_rep and not rev_src:
            rev_src = "yf_income_stmt"
    rev_beat = _beat_str(rev_rep, rev_est) if rev_est is not None else ""

    def _f(v):
        if v is None: return ""
        try: return f"{float(v):.4f}"
        except: return str(v)

    return {
        "date":        date,
        "eps_reported": _f(eps_rep),
        "eps_estimate": _f(eps_est),
        "eps_beat":    eps_beat,
        "eps_source":  eps_src,
        "rev_reported": _f(rev_rep) if rev_rep is not None else "",
        "rev_estimate": _f(rev_est) if rev_est is not None else "",
        "rev_beat":    rev_beat,
        "rev_source":  rev_src,
    }


def _fwd_quarter(entry: dict) -> dict:
    """Extract the nearest upcoming (forward) quarter from forward_estimates."""
    import datetime as _dt
    today = _dt.date.today().isoformat()

    # First try forward_estimates block (gc_engine v0.6.5+)
    fwd = entry.get("forward_estimates") or {}
    if fwd:
        date      = (fwd.get("date") or "")[:10]
        eps_est   = fwd.get("eps_estimate")
        eps_src   = fwd.get("eps_source") or "fmp"
        rev_est   = fwd.get("revenue_estimate")
        rev_src   = fwd.get("revenue_source") or ("fmp" if rev_est else "")
        def _f(v):
            if v is None: return ""
            try: return f"{float(v):.4f}"
            except: return str(v)
        return {
            "date":        date,
            "eps_estimate": _f(eps_est),
            "eps_source":  eps_src,
            "rev_estimate": _f(rev_est) if rev_est is not None else "",
            "rev_source":  rev_src,
        }

    # Fallback: scan earnings_dates for the nearest future quarter with an eps_estimate
    eds = entry.get("earnings_dates") or []
    future = sorted(
        [e for e in eds if (e.get("date") or "") > today and e.get("eps_estimate") is not None],
        key=lambda e: e["date"]
    )
    if not future:
        return {}
    q = future[0]
    def _f(v):
        if v is None: return ""
        try: return f"{float(v):.4f}"
        except: return str(v)
    return {
        "date":        (q.get("date") or "")[:10],
        "eps_estimate": _f(q.get("eps_estimate")),
        "eps_source":  q.get("_eps_est_source") or "yfinance",
        "rev_estimate": _f(q.get("revenue_estimate")) if q.get("revenue_estimate") is not None else "",
        "rev_source":  q.get("_rev_source") or ("fmp" if q.get("revenue_estimate") else ""),
    }


def _read_csv_tickers(path):
    """Read tickers from a classification CSV and normalize them via universe._normalize_ticker."""
    if not path.exists():
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("Ticker") or row.get("ticker") or "").strip()
            if t:
                # v4: normalize immediately so CSV-sourced tickers match gc_state keys
                t = _normalize_ticker(t)
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

        # ── Company name ────────────────────────────────────────────
        company = info.get("longName") or info.get("shortName") or ""

        # ── mcap ───────────────────────────────────────────────────
        # Prefer _mcap_usd (FX-converted by gc_engine v0.8.0)
        mcap_usd   = entry.get("_mcap_usd")
        mcap_local = info.get("market_cap") or entry.get("_last_known_mcap")
        mcap_currency = (info.get("currency") or "").upper()
        mcap_source = ""
        if mcap_usd:
            mcap_source = entry.get("_mcap_source") or "yf"
        elif mcap_local:
            mcap_source = entry.get("_mcap_source") or "yf"

        # passes_mcap: ? when unknown, N only when we KNOW it's below floor
        if mcap_usd and mcap_usd > 0:
            passes_mcap = "Y" if mcap_usd >= MIN_MCAP_USD else "N"
        else:
            passes_mcap = "?"

        # ── Status ──────────────────────────────────────────────────
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

        # ── Earnings dates — split historical vs forward ─────────────
        import datetime as _dt
        today = _dt.date.today().isoformat()
        all_eds = entry.get("earnings_dates") or []
        hist_eds = sorted(
            [e for e in all_eds if e.get("eps_reported") is not None
             and (e.get("date") or "") <= today],
            key=lambda e: e.get("date", ""), reverse=True
        )

        # Build quarterly_revenue lookup by year-month for rev matching
        qr_by_ym = {}
        for qr in (entry.get("quarterly_revenue") or []):
            d = (qr.get("date") or "")[:7]
            if d:
                qr_by_ym[d] = qr

        def _match_rev(ed_date):
            """Find best-matching rev row within ±1 month of earnings date."""
            d = (ed_date or "")[:7]
            if not d: return None
            if d in qr_by_ym: return qr_by_ym[d]
            y, m = int(d[:4]), int(d[5:7])
            for delta in [-1, 1, -2, 2]:
                nm = m + delta; ny = y + (nm - 1) // 12; nm = ((nm - 1) % 12) + 1
                key = f"{ny:04d}-{nm:02d}"
                if key in qr_by_ym: return qr_by_ym[key]
            return None

        # 4 most recent historical quarters
        q_cols = {}
        for qi, ed in enumerate(hist_eds[:4], 1):
            qdata = _quarter_row(ed, _match_rev(ed.get("date")))
            prefix = f"q{qi}_"
            q_cols[prefix + "date"]         = qdata.get("date", "")
            q_cols[prefix + "eps_reported"] = qdata.get("eps_reported", "")
            q_cols[prefix + "eps_estimate"] = qdata.get("eps_estimate", "")
            q_cols[prefix + "eps_beat"]     = qdata.get("eps_beat", "")
            q_cols[prefix + "eps_source"]   = qdata.get("eps_source", "")
            q_cols[prefix + "rev_reported"] = qdata.get("rev_reported", "")
            q_cols[prefix + "rev_estimate"] = qdata.get("rev_estimate", "")
            q_cols[prefix + "rev_beat"]     = qdata.get("rev_beat", "")
            q_cols[prefix + "rev_source"]   = qdata.get("rev_source", "")
        # Fill missing quarters with empty strings
        for qi in range(len(hist_eds[:4]) + 1, 5):
            prefix = f"q{qi}_"
            for sf in ["date","eps_reported","eps_estimate","eps_beat","eps_source",
                       "rev_reported","rev_estimate","rev_beat","rev_source"]:
                q_cols[prefix + sf] = ""

        # Forward quarter
        fwd = _fwd_quarter(entry)
        fwd_cols = {
            "fwd_date":         fwd.get("date", ""),
            "fwd_eps_estimate": fwd.get("eps_estimate", ""),
            "fwd_eps_source":   fwd.get("eps_source", ""),
            "fwd_rev_estimate": fwd.get("rev_estimate", ""),
            "fwd_rev_source":   fwd.get("rev_source", ""),
        }

        # ── has_* flags ─────────────────────────────────────────────
        has_eps_history = any(e.get("eps_reported") is not None for e in all_eds
                              if (e.get("date") or "") <= today)
        has_rev_history = len(entry.get("quarterly_revenue") or []) >= 4
        has_fwd_eps     = bool(fwd_cols["fwd_eps_estimate"])
        has_fwd_rev     = bool(fwd_cols["fwd_rev_estimate"])
        data_gap        = not (has_eps_history or has_rev_history
                               or info.get("revenue_growth") or info.get("market_cap"))

        rows.append({
            "ticker":          ticker,
            "company":         company,
            "country":         _country(ticker),
            "exchange":        _exch(ticker),
            "source_csvs":     ",".join(sorted(s for s in sources if s != "state_only")),
            "in_em_csv":       "Y" if "em" in sources else "",
            "market_cap_usd_b": _fmt_b(mcap_usd),
            "mcap_local_b":    _fmt_b(mcap_local),
            "mcap_currency":   mcap_currency,
            "mcap_source":     mcap_source,
            "passes_mcap":     passes_mcap,
            "status":          status,
            "no_data_runs":    str(ndr) if ndr else "",
            "ghost_risk":      ghost_risk,
            "has_eps_history": "Y" if has_eps_history else "",
            "has_rev_history": "Y" if has_rev_history else "",
            "has_fwd_eps":     "Y" if has_fwd_eps else "",
            "has_fwd_rev":     "Y" if has_fwd_rev else "",
            "data_gap":        "Y" if data_gap else "",
            **q_cols,
            **fwd_cols,
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
    print(f"\n[export] Market cap filter (unified $2B USD floor):")
    print(f"  Passes floor:   {mc_yes:,}")
    print(f"  Below floor:    {mc_no:,}")
    print(f"  Unknown mcap:   {mc_unk:,}  ← missing _mcap_usd in gc_state")

    norm_note = "(normalization active)" if _HAS_UNIVERSE else "(normalization DISABLED — universe.py not found)"
    print(f"\n[export] Ticker normalization: {norm_note}")

    active_rows = [r for r in rows if r["status"] == "active"]
    print(f"  has_eps_history: {sum(1 for r in active_rows if r['has_eps_history']=='Y'):,}")
    print(f"  has_rev_history: {sum(1 for r in active_rows if r['has_rev_history']=='Y'):,}")
    print(f"  has_fwd_eps:     {sum(1 for r in active_rows if r['has_fwd_eps']=='Y'):,}")
    print(f"  has_fwd_rev:     {sum(1 for r in active_rows if r['has_fwd_rev']=='Y'):,}")

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
