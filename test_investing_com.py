#!/usr/bin/env python3
"""
test_investing_com.py — Test investing.com revenue coverage on Nasdaq 100.

Tests three things per ticker:
  1. Revenue ESTIMATES (analyst consensus)  — needed for real beat/miss scoring
  2. Revenue ACTUALS from earnings page     — fills earnings_dates.revenue_reported
  3. Revenue ACTUALS for quarterly_revenue  — extends yfinance's 5Q cap to ~18Q

Usage:
    python test_investing_com.py                  # full Nasdaq 100
    python test_investing_com.py --limit 10       # quick smoke-test
    python test_investing_com.py --ticker NVDA    # single ticker debug
    python test_investing_com.py --verbose        # show per-row detail

Place this file in the same directory as gc_engine.py.
"""

import argparse, time, sys, json, datetime
from typing import Dict, List, Optional

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
    """20 stub rows, one per quarter going back 5 years. All estimates None."""
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

def build_stub_quarterly_revenue(n_quarters=5):
    """Simulate yfinance's 5Q hard cap on revenue actuals."""
    rows = []
    today = datetime.date.today()
    for q in range(n_quarters):
        months_back = q * 3 + 2
        year = today.year
        month = today.month - months_back
        while month <= 0:
            month += 12; year -= 1
        rows.append({
            "date": f"{year:04d}-{month:02d}-01",
            "revenue": 1.0e9,
            "revenue_yoy_growth": None,
            "revenue_source": "yfinance_stub",
        })
    return rows

def run_test(tickers, verbose=False):
    try:
        from gc_engine import enrich_estimates_investing_com
    except ImportError:
        print("ERROR: gc_engine.py not found. Run from same directory.")
        sys.exit(1)

    results = []
    total = len(tickers)
    YFINANCE_CAP = 5

    print(f"\nTesting investing.com on {total} Nasdaq 100 tickers")
    print(f"Stub: 20 earnings_dates rows | {YFINANCE_CAP}Q quarterly_revenue (simulates yfinance cap)")
    print(f"Goal: fill Rev Estimate, Rev Actual, extend quarterly_revenue beyond {YFINANCE_CAP}Q\n")
    print(f"{'#':>3}  {'Ticker':<6}  {'RevEst(Q1)':>11}  {'RevAct(Q1)':>11}  {'Est/Act Qs':>10}  {'QRev Depth':>12}  {'Time':>5}")
    print("-" * 85)

    for i, ticker in enumerate(tickers, 1):
        stub_ed = build_stub_earnings_dates(20)
        stub_qr = build_stub_quarterly_revenue(YFINANCE_CAP)
        t0 = time.time()
        try:
            filled = enrich_estimates_investing_com(stub_ed, ticker, quarterly_revenue=stub_qr)
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"ticker": ticker, "filled": 0, "error": str(e), "elapsed": elapsed,
                             "rev_est_count": 0, "rev_act_count": 0, "eps_est_count": 0,
                             "qr_depth_before": YFINANCE_CAP, "qr_depth_after": YFINANCE_CAP})
            print(f"{i:>3}  {ticker:<6}  ERROR: {str(e)[:50]}")
            continue

        elapsed = time.time() - t0
        rev_ests = [r["revenue_estimate"] for r in stub_ed if r.get("revenue_estimate") is not None]
        rev_acts = [r["revenue_reported"]  for r in stub_ed if r.get("revenue_reported")  is not None]
        eps_ests = [r["eps_estimate"]       for r in stub_ed if r.get("eps_estimate")       is not None]
        qr_after = len([r for r in stub_qr if r.get("revenue") is not None])

        result = {
            "ticker": ticker, "filled": filled, "error": None, "elapsed": elapsed,
            "rev_est_count": len(rev_ests), "rev_act_count": len(rev_acts),
            "eps_est_count": len(eps_ests),
            "rev_est_latest": rev_ests[0] if rev_ests else None,
            "rev_act_latest": rev_acts[0] if rev_acts else None,
            "eps_est_latest": eps_ests[0] if eps_ests else None,
            "qr_depth_before": YFINANCE_CAP, "qr_depth_after": qr_after,
        }
        results.append(result)

        depth_str = (f"{YFINANCE_CAP}Q→{qr_after}Q ✓" if qr_after > YFINANCE_CAP
                     else f"{qr_after}Q")
        print(f"{i:>3}  {ticker:<6}  "
              f"{_fmt_rev(result['rev_est_latest']):>11}  "
              f"{_fmt_rev(result['rev_act_latest']):>11}  "
              f"{len(rev_ests):>3}est/{len(rev_acts):<3}act  "
              f"{depth_str:>12}  "
              f"{elapsed:>4.1f}s")

        if verbose and (rev_ests or rev_acts or qr_after > YFINANCE_CAP):
            print(f"  earnings_dates filled rows:")
            for r in stub_ed:
                if r.get("revenue_estimate") or r.get("revenue_reported"):
                    print(f"    {r['date'][:7]}  est={_fmt_rev(r.get('revenue_estimate'))}  "
                          f"act={_fmt_rev(r.get('revenue_reported'))}  eps={r.get('eps_estimate')}")
            ic_ext = [r for r in stub_qr if r.get("revenue_source") == "investing_com"]
            if ic_ext:
                print(f"  quarterly_revenue extended ({len(ic_ext)} new quarters from IC):")
                for r in ic_ext[:10]:
                    yoy = f"{r['revenue_yoy_growth']:+.1f}%" if r.get("revenue_yoy_growth") else "–"
                    print(f"    {r['date'][:7]}  {_fmt_rev(r.get('revenue'))}  YoY={yoy}")

        time.sleep(1.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok       = [r for r in results if not r["error"]]
    got_est  = [r for r in ok if r["rev_est_count"] > 0]
    got_act  = [r for r in ok if r["rev_act_count"] > 0]
    got_eps  = [r for r in ok if r["eps_est_count"] > 0]
    got_ext  = [r for r in ok if r["qr_depth_after"] > r["qr_depth_before"]]
    errors   = [r for r in results if r["error"]]
    avg_time = sum(r["elapsed"] for r in results) / max(len(results), 1)
    avg_est_q = sum(r["rev_est_count"] for r in got_est) / max(len(got_est), 1)
    avg_act_q = sum(r["rev_act_count"] for r in got_act) / max(len(got_act), 1)
    avg_depth_after = sum(r["qr_depth_after"] for r in ok) / max(len(ok), 1)

    print(f"\n{'='*85}")
    print(f"RESULTS — investing.com Nasdaq 100 ({total} tickers)\n")
    print(f"  Revenue Estimate filled:       {len(got_est):>4}/{total}  ({len(got_est)*100//total}%)   avg {avg_est_q:.1f} quarters/ticker")
    print(f"  Revenue Actual filled (ed):    {len(got_act):>4}/{total}  ({len(got_act)*100//total}%)   avg {avg_act_q:.1f} quarters/ticker")
    print(f"  EPS Estimate filled:           {len(got_eps):>4}/{total}  ({len(got_eps)*100//total}%)")
    print(f"  quarterly_revenue extended:    {len(got_ext):>4}/{total}  ({len(got_ext)*100//total}%)")
    print(f"  depth: yfinance stub={YFINANCE_CAP}Q → after IC avg={avg_depth_after:.1f}Q")
    print(f"  Errors:                        {len(errors):>4}/{total}")
    print(f"  Avg time per ticker:           {avg_time:.1f}s")
    print(f"  Full-universe estimate (2558): ~{int(avg_time * 2558 / 60)} min")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors[:10]:
            print(f"  {r['ticker']}: {r['error']}")
    silent = [r for r in ok if r["filled"] == 0]
    if silent:
        syms = [r["ticker"] for r in silent]
        print(f"\nNo data — silent ({len(silent)}): {', '.join(syms[:25])}"
              + (f" ... +{len(syms)-25}" if len(syms) > 25 else ""))

    output_path = "investing_com_test_results.json"
    save = [{k: v for k, v in r.items()} for r in results]
    with open(output_path, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nFull results → {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",   type=int, default=0)
    ap.add_argument("--ticker",  type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tickers = ([args.ticker.upper()] if args.ticker
               else NASDAQ_100[:args.limit] if args.limit > 0
               else NASDAQ_100)
    run_test(tickers, verbose=args.verbose)
