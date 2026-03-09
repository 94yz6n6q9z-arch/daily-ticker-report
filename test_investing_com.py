#!/usr/bin/env python3
"""
test_investing_com.py — Test investing.com revenue coverage on full universe.

Can run against:
  - gc_state.json (full 2558-ticker universe) — default
  - Nasdaq 100 hardcoded list (--nasdaq)
  - A single ticker (--ticker NVDA)

Usage:
    python3 test_investing_com.py                   # full universe from gc_state.json
    python3 test_investing_com.py --limit 100       # first 100 from state
    python3 test_investing_com.py --nasdaq          # Nasdaq 100 only
    python3 test_investing_com.py --ticker NVDA     # single ticker debug
    python3 test_investing_com.py --verbose         # show per-row detail
    python3 test_investing_com.py --missed          # only tickers with no IC data yet

Place this file in the same directory as gc_engine.py and gc_state.json.
"""

import argparse, time, sys, json, datetime
from pathlib import Path
from typing import List, Dict, Optional

NASDAQ_100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AZN","AMD","PEP","LIN","QCOM","INTU","AMAT","CSCO",
    "AMGN","TXN","CMCSA","BKNG","MU","HON","PANW","GILD","SBUX","ADP",
    "VRTX","REGN","ADI","MELI","KLAC","LRCX","CRWD","ISRG","MDLZ","CTAS",
    "SNPS","CDNS","MNST","MRVL","CEG","ROP","PYPL","WDAY","ORLY","PCAR",
    "FTNT","NXPI","CHTR","ABNB","MCHP","PAYX","FAST","ROST","DDOG","ZS",
    "MAR","ODFL","TTD","ON","CPRT","KDP","EXC","IDXX","VRSK","XEL",
    "TEAM","ANSS","FANG","CTSH","BIIB","DLTR","KHC","WBD","ILMN","SIRI",
    "TTWO","MTCH","ZM","LCID","NTES","PDD","RIVN","ALGN","ENPH","DXCM",
    "MDB","GFS","GEHC","SMCI","APP","AXON","DASH","RBLX","COIN","PLTR",
]

def _fmt_rev(v):
    if v is None: return "–"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:.0f}"

def build_stub_earnings_dates(n_quarters=20):
    rows = []
    today = datetime.date.today()
    for q in range(n_quarters):
        months_back = q * 3 + 1
        year = today.year
        month = today.month - months_back
        while month <= 0:
            month += 12; year -= 1
        rows.append({
            "date": f"{year:04d}-{month:02d}-15",
            "eps_estimate": None, "eps_reported": 1.0,
            "eps_surprise_pct": None,
            "revenue_estimate": None, "revenue_reported": None,
            "_method": "test_stub",
        })
    return rows

def load_tickers_from_state(state_file="gc_state.json", missed_only=False) -> List[str]:
    """Load tickers from gc_state.json, skipping inactive ones."""
    if not Path(state_file).exists():
        print(f"ERROR: {state_file} not found. Run from repo root.")
        sys.exit(1)
    with open(state_file) as f:
        state = json.load(f)

    tickers = []
    for ticker, data in state.items():
        if data.get("inactive"):
            continue
        if missed_only:
            # Only include tickers that have NO revenue estimates yet
            has_rev_est = any(
                r.get("revenue_estimate") is not None
                for r in data.get("earnings_dates", [])
                if r.get("eps_reported") is not None
            )
            if has_rev_est:
                continue
        # Prioritise by data richness (most EPS + revenue data first)
        score = (len([r for r in data.get("earnings_dates", []) if r.get("eps_reported") is not None])
                 + len(data.get("quarterly_revenue", [])) * 2)
        tickers.append((score, ticker))

    tickers.sort(key=lambda x: -x[0])
    return [t for _, t in tickers]

def run_test(tickers: List[str], verbose=False):
    try:
        from gc_engine import enrich_estimates_investing_com
    except ImportError:
        print("ERROR: gc_engine.py not found. Run from same directory.")
        sys.exit(1)

    results = []
    total = len(tickers)

    print(f"\nTesting investing.com on {total} tickers")
    print(f"Stub: 20 earnings_dates rows per ticker\n")
    print(f"{'#':>5}  {'Ticker':<10}  {'RevEst':>10}  {'RevAct':>10}  {'Qs':>8}  {'Time':>5}  Status")
    print("-" * 75)

    got_data = 0
    got_429  = 0

    for i, ticker in enumerate(tickers, 1):
        stub_ed = build_stub_earnings_dates(20)
        t0 = time.time()
        try:
            filled = enrich_estimates_investing_com(stub_ed, ticker)
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"ticker": ticker, "filled": 0, "error": str(e),
                             "elapsed": elapsed, "rev_est_count": 0, "rev_act_count": 0})
            print(f"{i:>5}  {ticker:<10}  ERROR: {str(e)[:40]}")
            continue

        elapsed = time.time() - t0
        rev_ests = [r["revenue_estimate"] for r in stub_ed if r.get("revenue_estimate") is not None]
        rev_acts = [r["revenue_reported"]  for r in stub_ed if r.get("revenue_reported")  is not None]
        eps_ests = [r["eps_estimate"]       for r in stub_ed if r.get("eps_estimate")       is not None]

        # Detect rate-limited (10s+ with no data = 429 triggered backoff)
        was_429 = elapsed > 8.0 and not rev_ests and not rev_acts

        result = {
            "ticker": ticker, "filled": filled, "error": None, "elapsed": elapsed,
            "rev_est_count": len(rev_ests), "rev_act_count": len(rev_acts),
            "eps_est_count": len(eps_ests),
            "rev_est_latest": rev_ests[0] if rev_ests else None,
            "rev_act_latest": rev_acts[0] if rev_acts else None,
            "was_429": was_429,
        }
        results.append(result)

        if rev_ests or rev_acts:
            got_data += 1
            status = "✓"
        elif was_429:
            got_429 += 1
            status = "429"
        else:
            status = "–"

        # Only print every ticker if verbose or has data; otherwise print every 10
        if verbose or rev_ests or rev_acts or was_429 or i % 10 == 0:
            print(f"{i:>5}  {ticker:<10}  "
                  f"{_fmt_rev(result['rev_est_latest']):>10}  "
                  f"{_fmt_rev(result['rev_act_latest']):>10}  "
                  f"{len(rev_ests):>3}e/{len(rev_acts):<3}a  "
                  f"{elapsed:>4.1f}s  {status}")

            if verbose and (rev_ests or rev_acts):
                for r in stub_ed:
                    if r.get("revenue_estimate") or r.get("revenue_reported"):
                        print(f"         {r['date'][:7]}  "
                              f"est={_fmt_rev(r.get('revenue_estimate'))}  "
                              f"act={_fmt_rev(r.get('revenue_reported'))}")

        time.sleep(3.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if not r["error"]]
    got_est = [r for r in ok if r["rev_est_count"] > 0]
    got_act = [r for r in ok if r["rev_act_count"] > 0]
    got_eps = [r for r in ok if r["eps_est_count"] > 0]
    errors  = [r for r in results if r["error"]]
    rate_limited = [r for r in ok if r["was_429"]]
    silent  = [r for r in ok if r["filled"] == 0 and not r["was_429"]]
    avg_time = sum(r["elapsed"] for r in results) / max(len(results), 1)
    avg_est_q = sum(r["rev_est_count"] for r in got_est) / max(len(got_est), 1)

    print(f"\n{'='*75}")
    print(f"RESULTS — investing.com full universe ({total} tickers)\n")
    print(f"  Revenue Estimate filled:    {len(got_est):>5}/{total}  ({len(got_est)*100//total}%)  avg {avg_est_q:.1f}Q/ticker")
    print(f"  Revenue Actual filled:      {len(got_act):>5}/{total}  ({len(got_act)*100//total}%)")
    print(f"  EPS Estimate filled:        {len(got_eps):>5}/{total}  ({len(got_eps)*100//total}%)")
    print(f"  Rate limited (429):         {len(rate_limited):>5}/{total}  (increase sleep if high)")
    print(f"  No data / not on IC:        {len(silent):>5}/{total}")
    print(f"  Errors:                     {len(errors):>5}/{total}")
    print(f"  Avg time per ticker:        {avg_time:.1f}s")

    # Save full results
    output_path = "investing_com_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results → {output_path}")

    # Save just the "got data" list for easy reference
    hits_path = "investing_com_hits.json"
    hits = {r["ticker"]: {"rev_est_qs": r["rev_est_count"], "rev_act_qs": r["rev_act_count"]}
            for r in got_est}
    with open(hits_path, "w") as f:
        json.dump(hits, f, indent=2)
    print(f"Tickers with data → {hits_path}")

    # Save the miss list (no data, not 429 — truly not on IC)
    misses_path = "investing_com_misses.json"
    misses = [r["ticker"] for r in silent]
    with open(misses_path, "w") as f:
        json.dump(misses, f, indent=2)
    print(f"Tickers with no IC data → {misses_path}  ({len(misses)} tickers)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",   type=int, default=0, help="Test only first N tickers")
    ap.add_argument("--ticker",  type=str, default="", help="Single ticker")
    ap.add_argument("--nasdaq",  action="store_true", help="Use Nasdaq 100 list instead of state")
    ap.add_argument("--missed",  action="store_true", help="Only tickers with no rev estimates yet")
    ap.add_argument("--verbose", action="store_true", help="Show per-row detail")
    ap.add_argument("--sleep",   type=float, default=3.0, help="Sleep between tickers (default 3s)")
    args = ap.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.nasdaq:
        tickers = NASDAQ_100[:args.limit] if args.limit else NASDAQ_100
    else:
        tickers = load_tickers_from_state(missed_only=args.missed)
        if args.limit:
            tickers = tickers[:args.limit]

    run_test(tickers, verbose=args.verbose)
