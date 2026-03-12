#!/usr/bin/env python3
"""
export_universe.py  —  Universe review export  v3
==================================================
Produces universe_review.csv with per-quarter data detail.

Key additions vs v2:
  Last 4 historical quarters (q1=most recent, q2-q4 older):
    q{n}_date, q{n}_eps_reported, q{n}_eps_estimate, q{n}_eps_beat, q{n}_eps_source,
    q{n}_rev_reported, q{n}_rev_estimate, q{n}_rev_beat, q{n}_rev_source

  1 forward quarter (nearest upcoming earnings):
    fwd_date, fwd_eps_estimate, fwd_eps_source, fwd_rev_estimate, fwd_rev_source

Design rules:
  - Historical vs forward STRICTLY separated — never mixed
  - Stars use ONLY historical reported quarters (gc_engine already enforces this)
  - Forward estimates have no beat/miss column (no actuals to compare against)
  - mcap shown in USD (_mcap_usd from gc_engine v0.8.0+, FX-converted)
  - passes_mcap=? when mcap unknown (only N when KNOWN below floor)
  - Universal $2B USD floor (was $2B US/EU + $5B EM)
"""

import argparse, csv, json, math, datetime as dt
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR   = BASE_DIR / "docs"

DEFAULT_STATE = DOCS_DIR / "gc_state.json"
DEFAULT_OUT   = BASE_DIR / "universe_review.csv"
MIN_MCAP_USD  = 2_000_000_000   # $2B USD universal floor

EXCHANGE_COUNTRY = {
    "US":"United States","TO":"Canada","SA":"Brazil","MX":"Mexico",
    "SN":"Chile","CL":"Colombia","CA":"Egypt",
    "L":"United Kingdom","DE":"Germany","PA":"France","AS":"Netherlands",
    "MI":"Italy","MC":"Spain","ST":"Sweden","OL":"Norway","HE":"Finland",
    "CO":"Denmark","LS":"Portugal","BR":"Belgium","IR":"Ireland",
    "SW":"Switzerland","WA":"Poland","VI":"Austria","AT":"Greece",
    "PR":"Czech Republic","BD":"Hungary",
    "SR":"Saudi Arabia","QA":"Qatar","KW":"Kuwait",
    "AE":"UAE","AD":"UAE Abu Dhabi (legacy)","DU":"UAE Dubai (legacy)",
    "JO":"South Africa","TA":"Israel",
    "T":"Japan","TW":"Taiwan","TWO":"Taiwan (Gretai)","HK":"Hong Kong",
    "SS":"China-SH","SZ":"China-SZ","KS":"South Korea","KQ":"South Korea (KOSDAQ)",
    "NS":"India (NSE)","BO":"India (BSE)","SI":"Singapore",
    "AX":"Australia","NZ":"New Zealand",
    "KL":"Malaysia (dead)","PS":"Philippines (dead)",
    "JK":"Indonesia","BK":"Thailand","IS":"Turkey",
}

CSV_DEFS = [("world","msci_world_classification.csv"),("em","msci_em_classification.csv")]


def _quarter_fieldnames(n):
    p = f"q{n}_"
    return [f"{p}date",f"{p}eps_reported",f"{p}eps_estimate",f"{p}eps_beat",f"{p}eps_source",
            f"{p}rev_reported",f"{p}rev_estimate",f"{p}rev_beat",f"{p}rev_source"]

FIELDNAMES = (
    ["ticker","company","country","exchange","source_csvs","in_em_csv",
     "market_cap_usd_b","mcap_local_b","mcap_currency","mcap_source","passes_mcap",
     "status","no_data_runs","ghost_risk",
     "has_eps_history","has_rev_history","has_fwd_eps","has_fwd_rev","data_gap"] +
    _quarter_fieldnames(1) + _quarter_fieldnames(2) +
    _quarter_fieldnames(3) + _quarter_fieldnames(4) +
    ["fwd_date","fwd_eps_estimate","fwd_eps_source","fwd_rev_estimate","fwd_rev_source",
     "inactive_reason","inactive_since","fetched_at","error"]
)


def _exch(t):
    return t.rsplit(".",1)[-1] if "." in t else "US"

def _country(t):
    return EXCHANGE_COUNTRY.get(_exch(t), f"Unknown ({_exch(t)})")

def _fmt_b(v):
    try:
        f = float(v)
        if math.isfinite(f) and f > 0:
            return f"{f/1e9:.2f}"
    except (TypeError, ValueError):
        pass
    return ""

def _safe_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _read_csv_tickers(path):
    if not path.exists():
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("Ticker") or row.get("ticker") or "").strip()
            c = (row.get("Company") or row.get("company") or row.get("Name") or "").strip()
            if t:
                result.append((t, c))
    return result

def _eps_beat_label(eps_rep, eps_est, surp_pct):
    if eps_rep is None:
        return ""
    if surp_pct is not None:
        s = float(surp_pct)
        return "match" if abs(s) <= 0.5 else ("beat" if s > 0 else "miss")
    if eps_est is not None:
        diff = float(eps_rep) - float(eps_est)
        return "match" if abs(diff) <= 0.01 else ("beat" if diff > 0 else "miss")
    return ""

def _rev_beat_label(rev_rep, rev_est):
    if rev_rep is None or rev_est is None:
        return ""
    r_rep, r_est = float(rev_rep), float(rev_est)
    if r_est <= 0:
        return ""
    ratio = (r_rep - r_est) / r_est
    return "match" if abs(ratio) <= 0.005 else ("beat" if ratio > 0 else "miss")

def _extract_quarters(entry):
    """Return (past_quarters[:4], fwd_quarter|None).
    past = historical with eps_reported not None, date <= today, newest first.
    fwd  = nearest future entry with at least one estimate, no actuals yet.
    Stars use past ONLY. fwd is display-only.
    """
    today = dt.date.today().isoformat()
    dates = entry.get("earnings_dates", [])
    past = sorted(
        [d for d in dates
         if d.get("eps_reported") is not None
         and (d.get("date") or "") <= today],
        key=lambda r: r.get("date",""), reverse=True
    )
    future = sorted(
        [d for d in dates
         if d.get("eps_reported") is None
         and (d.get("eps_estimate") is not None or d.get("revenue_estimate") is not None)
         and ((d.get("date") or "") > today or d.get("_is_forward"))],
        key=lambda r: r.get("date","")
    )
    return past[:4], (future[0] if future else None)

def _quarter_cols(q, n):
    """Return dict of q{n}_* columns for one historical quarter."""
    prefix = f"q{n}_"
    if not q:
        return {f: "" for f in _quarter_fieldnames(n)}
    eps_rep  = _safe_float(q.get("eps_reported"))
    eps_est  = _safe_float(q.get("eps_estimate"))
    eps_surp = _safe_float(q.get("eps_surprise_pct"))
    rev_rep  = _safe_float(q.get("revenue_reported"))
    rev_est  = _safe_float(q.get("revenue_estimate"))
    return {
        f"{prefix}date":         (q.get("date") or "")[:10],
        f"{prefix}eps_reported": "" if eps_rep is None else f"{eps_rep:.4f}",
        f"{prefix}eps_estimate": "" if eps_est is None else f"{eps_est:.4f}",
        f"{prefix}eps_beat":     _eps_beat_label(eps_rep, eps_est, eps_surp),
        f"{prefix}eps_source":   q.get("_eps_est_source") or "",
        f"{prefix}rev_reported": _fmt_b(rev_rep) if rev_rep is not None else "",
        f"{prefix}rev_estimate": _fmt_b(rev_est) if rev_est is not None else "",
        f"{prefix}rev_beat":     _rev_beat_label(rev_rep, rev_est),
        f"{prefix}rev_source":   q.get("_rev_est_source") or q.get("_rev_act_source") or "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--out",   default=str(DEFAULT_OUT))
    args = ap.parse_args()

    cache = {}
    if Path(args.state).exists():
        print(f"[export] Loading {args.state} …")
        with open(args.state, "r", encoding="utf-8") as f:
            state = json.load(f)
        cache = state.get("earnings_cache", {})
        print(f"[export] gc_state entries: {len(cache):,}")
    else:
        print(f"[export] WARNING: {args.state} not found")

    csv_sources   = defaultdict(set)
    csv_companies = {}
    for name, fname in CSV_DEFS:
        for t, comp in _read_csv_tickers(CONFIG_DIR / fname):
            csv_sources[t].add(name)
            if comp and t not in csv_companies:
                csv_companies[t] = comp

    for t in cache:
        if t not in csv_sources:
            csv_sources[t].add("state_only")

    all_tickers = sorted(csv_sources.keys())
    print(f"[export] Unique tickers: {len(all_tickers):,}")

    rows = []
    for ticker in all_tickers:
        entry  = cache.get(ticker, {})
        info   = entry.get("info") or {}
        sources = csv_sources[ticker]

        # mcap — prefer _mcap_usd (FX-converted), fall back to local
        mcap_usd   = _safe_float(entry.get("_mcap_usd"))
        mcap_local = _safe_float(info.get("market_cap") or entry.get("_last_known_mcap"))
        mcap_cur   = (info.get("currency") or entry.get("_mcap_currency") or "")
        mcap_src   = entry.get("_mcap_source", "yf" if info.get("market_cap") else "")

        mcap_usd_b   = _fmt_b(mcap_usd)   if (mcap_usd   and mcap_usd   > 0) else ""
        mcap_local_b = _fmt_b(mcap_local) if (mcap_local and mcap_local > 0) else ""

        # passes_mcap: only N when KNOWN below floor; ? when unknown
        if mcap_usd and mcap_usd > 0:
            passes_mcap = "Y" if mcap_usd >= MIN_MCAP_USD else "N"
        elif entry.get("below_min_mcap"):
            passes_mcap = "N"
        else:
            passes_mcap = "?"

        # status
        if entry.get("inactive"):
            ir    = entry.get("inactive_reason", "")
            since = (entry.get("inactive_since") or "")[:10]
            if any(x in ir for x in ["no_price_no_financials","known_dead","dead_market"]):
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

        past_qs, fwd = _extract_quarters(entry)

        has_eps_hist = any(q.get("eps_reported") is not None for q in past_qs)
        has_rev_hist = len(entry.get("quarterly_revenue", [])) >= 1
        has_fwd_eps  = fwd is not None and fwd.get("eps_estimate") is not None
        has_fwd_rev  = fwd is not None and fwd.get("revenue_estimate") is not None
        data_gap     = not (has_eps_hist or has_rev_hist)

        q_data = {}
        for n in range(1, 5):
            q_data.update(_quarter_cols(past_qs[n-1] if n <= len(past_qs) else None, n))

        if fwd:
            fe = _safe_float(fwd.get("eps_estimate"))
            fr = _safe_float(fwd.get("revenue_estimate"))
            fwd_cols = {
                "fwd_date":         (fwd.get("date") or "")[:10],
                "fwd_eps_estimate": "" if fe is None else f"{fe:.4f}",
                "fwd_eps_source":   fwd.get("_eps_est_source") or "",
                "fwd_rev_estimate": _fmt_b(fr) if fr else "",
                "fwd_rev_source":   fwd.get("_rev_est_source") or "",
            }
        else:
            fwd_cols = {k:"" for k in ["fwd_date","fwd_eps_estimate","fwd_eps_source",
                                        "fwd_rev_estimate","fwd_rev_source"]}

        rows.append({
            "ticker":           ticker,
            "company":          csv_companies.get(ticker, ""),
            "country":          _country(ticker),
            "exchange":         _exch(ticker),
            "source_csvs":      ",".join(sorted(sources)),
            "in_em_csv":        "Y" if "em" in sources else "",
            "market_cap_usd_b": mcap_usd_b,
            "mcap_local_b":     mcap_local_b,
            "mcap_currency":    mcap_cur,
            "mcap_source":      mcap_src,
            "passes_mcap":      passes_mcap,
            "status":           status,
            "no_data_runs":     str(ndr) if ndr else "",
            "ghost_risk":       ghost_risk,
            "has_eps_history":  "Y" if has_eps_hist else "",
            "has_rev_history":  "Y" if has_rev_hist else "",
            "has_fwd_eps":      "Y" if has_fwd_eps else "",
            "has_fwd_rev":      "Y" if has_fwd_rev else "",
            "data_gap":         "Y" if data_gap else "",
            **q_data,
            **fwd_cols,
            "inactive_reason":  entry.get("inactive_reason", ""),
            "inactive_since":   (entry.get("inactive_since") or "")[:10],
            "fetched_at":       (entry.get("fetched_at") or "")[:10],
            "error":            str(entry.get("error", ""))[:80],
        })

    rows.sort(key=lambda r: (r["country"], r["ticker"]))

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[export] Written {len(rows):,} rows → {out_path}")

    sc = Counter(r["status"] for r in rows)
    print(f"\n[export] Status breakdown:")
    for s, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:30s}: {n:5,}")

    mc_yes = sum(1 for r in rows if r["passes_mcap"] == "Y")
    mc_no  = sum(1 for r in rows if r["passes_mcap"] == "N")
    mc_unk = sum(1 for r in rows if r["passes_mcap"] == "?")
    print(f"\n[export] Market cap ($2B USD floor):")
    print(f"  Passes  (USD mcap known >= 2B) : {mc_yes:,}")
    print(f"  Below   (USD mcap known <  2B) : {mc_no:,}")
    print(f"  Unknown (no USD mcap available): {mc_unk:,}  ← run gc_engine v0.8.0+ to populate")

    he = sum(1 for r in rows if r["has_eps_history"] == "Y")
    hr = sum(1 for r in rows if r["has_rev_history"] == "Y")
    fe = sum(1 for r in rows if r["has_fwd_eps"]     == "Y")
    fr = sum(1 for r in rows if r["has_fwd_rev"]     == "Y")
    print(f"\n[export] Data coverage (2 separate dimensions — never mixed):")
    print(f"  Historical EPS (reported actuals): {he:,}")
    print(f"  Historical Rev (income statement): {hr:,}")
    print(f"  Forward EPS estimate             : {fe:,}  ← display only, not scored")
    print(f"  Forward Rev estimate             : {fr:,}  ← display only, not scored")
    print(f"\n  Stars use ONLY historical beats. fwd columns have no beat/miss field.")

    ca = Counter(r["country"] for r in rows if r["status"] == "active")
    print(f"\n[export] Active tickers by country (top 30):")
    for country, n in ca.most_common(30):
        print(f"  {country:40s}: {n:4,}")

    print(f"\n[export] Done.")


if __name__ == "__main__":
    main()
