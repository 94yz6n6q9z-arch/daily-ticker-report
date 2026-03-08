#!/usr/bin/env python3
"""
Finnhub Coverage Test
Tests key validity, symbol format, and actual earnings data return
for the exchanges where yfinance fails.

Run via GitHub Actions or locally:
    FINNHUB_API_KEY=your_key python3 test_finnhub.py
"""

import json, os, sys, time, urllib.parse, urllib.request
from collections import defaultdict

BASE = "https://finnhub.io/api/v1"
KEY  = os.environ.get("FINNHUB_API_KEY", "").strip()
if not KEY:
    print("FINNHUB_API_KEY not set."); sys.exit(1)

def get(path, params=None):
    params = {**(params or {}), "token": KEY}
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)

# ── Test 1: Key validity ──────────────────────────────────────────
print("\n-- Test 1: Key validity --")
data, err = get("/quote", {"symbol": "AAPL"})
if err:
    print(f"[FAIL] {err}"); sys.exit(1)
if data and data.get("c"):
    print(f"[OK  ] Key valid — AAPL quote: ${data['c']}")
else:
    print(f"[WARN] Unexpected response: {data}")

# ── Test 2: Earnings data for US (control) ────────────────────────
print("\n-- Test 2: US earnings (control) --")
data, err = get("/stock/earnings", {"symbol": "AAPL", "limit": 4})
if data and isinstance(data, list) and len(data) > 0:
    print(f"[OK  ] AAPL earnings: {len(data)} quarters found")
    print(f"       Latest: actual={data[0].get('actual')}  estimate={data[0].get('estimate')}  surprise={data[0].get('surprisePercent')}%")
else:
    print(f"[FAIL] No earnings data — {err}")

# ── Test 3: International symbol formats ─────────────────────────
# Finnhub uses EXCHANGE:SYMBOL format for international stocks.
# We test multiple candidate formats per exchange to find what works.
print("\n-- Test 3: International symbol format discovery --")
print("   Testing multiple formats per exchange to find what Finnhub accepts\n")

# Each entry: (yahoo_ticker, [format_candidates], exchange_label)
INTL_TESTS = [
    ("CIMB.KL",  ["KLSE:CIMB",  "MYX:CIMB",   "CIMB.KL"],  "Malaysia .KL"),
    ("EREGL.IS", ["BIST:EREGL", "IST:EREGL",   "EREGL.IS"], "Turkey .IS"),
    ("BDO.PS",   ["PSE:BDO",    "PHP:BDO",     "BDO.PS"],   "Philippines .PS"),
    ("FAB.AD",   ["ADX:FAB",    "ABU:FAB",     "FAB.AD"],   "Abu Dhabi .AD"),
    ("DIB.DU",   ["DFM:DIB",    "DXB:DIB",     "DIB.DU"],   "Dubai .DU"),
    ("7203.T",   ["TSE:7203",   "TYO:7203",    "7203.T"],   "Japan .T"),
    ("AZN.L",    ["LSE:AZN",    "LON:AZN",     "AZN.L"],    "UK .L"),
    ("CBA.AX",   ["ASX:CBA",    "AUS:CBA",     "CBA.AX"],   "Australia .AX"),
    ("MAYBANK.KL",["KLSE:MAYBANK","MYX:MAYBANK","MAYBANK.KL"],"Malaysia (2nd)"),
    ("THYAO.IS", ["BIST:THYAO", "IST:THYAO",   "THYAO.IS"], "Turkey (2nd)"),
]

working_formats = {}  # exchange -> best format template

for yahoo, candidates, label in INTL_TESTS:
    found_format = None
    found_data   = None
    for fmt in candidates:
        d, e = get("/stock/earnings", {"symbol": fmt, "limit": 2})
        time.sleep(0.3)
        if d and isinstance(d, list) and len(d) > 0 and d[0].get("actual") is not None:
            found_format = fmt
            found_data   = d
            break
        # Also try basic financials as fallback
        d2, e2 = get("/stock/metric", {"symbol": fmt, "metric": "all"})
        time.sleep(0.3)
        if d2 and d2.get("metric") and d2["metric"].get("revenueGrowthQuarterlyYoy") is not None:
            found_format = fmt + " (metrics only)"
            found_data   = d2
            break

    if found_format:
        quarters = len(found_data) if isinstance(found_data, list) else "metrics"
        print(f"  [OK  ] {yahoo:14s} -> {found_format:20s}  {label}  ({quarters} data points)")
        # Extract exchange prefix for template
        if ":" in found_format:
            exch_prefix = found_format.split(":")[0]
            yahoo_suffix = yahoo.rsplit(".", 1)[-1] if "." in yahoo else "US"
            working_formats[yahoo_suffix] = exch_prefix
    else:
        print(f"  [FAIL] {yahoo:14s} -> no working format found  {label}")
        print(f"         tried: {', '.join(candidates)}")

# ── Test 4: Earnings data quality for working formats ─────────────
print("\n-- Test 4: Earnings data quality --")
print("   Checking revenue + EPS beat data where formats work\n")

QUALITY_TESTS = []
for yahoo, candidates, label in INTL_TESTS[:8]:
    yahoo_suffix = yahoo.rsplit(".", 1)[-1] if "." in yahoo else "US"
    if yahoo_suffix in working_formats:
        base = yahoo.rsplit(".", 1)[0]
        fmt  = f"{working_formats[yahoo_suffix]}:{base}"
        QUALITY_TESTS.append((yahoo, fmt, label))

for yahoo, fmt, label in QUALITY_TESTS:
    eps_data, _   = get("/stock/earnings",         {"symbol": fmt, "limit": 8})
    time.sleep(0.2)
    fin_data, _   = get("/stock/financials-reported", {"symbol": fmt, "freq": "quarterly"})
    time.sleep(0.2)
    metric_data,_ = get("/stock/metric",           {"symbol": fmt, "metric": "all"})
    time.sleep(0.2)

    has_eps = isinstance(eps_data, list) and len(eps_data) > 0 and eps_data[0].get("actual") is not None
    has_fin = isinstance(fin_data, dict) and fin_data.get("data") and len(fin_data["data"]) > 0
    has_rev = isinstance(metric_data, dict) and metric_data.get("metric", {}).get("revenueGrowthQuarterlyYoy") is not None

    parts = []
    if has_eps: parts.append(f"EPS({len(eps_data)}Q)")
    if has_fin: parts.append(f"financials({len(fin_data['data'])}Q)")
    if has_rev: parts.append("rev_growth")

    status = "[OK  ]" if (has_eps or has_fin) else "[WARN]" if has_rev else "[FAIL]"
    detail = "  ".join(parts) if parts else "no earnings/revenue data"
    print(f"  {status} {yahoo:14s} ({fmt:20s})  {label:15s}  {detail}")

# ── Summary ───────────────────────────────────────────────────────
print("\n-- Summary --")
print(f"  Working exchange prefix mappings found:")
if working_formats:
    for suffix, prefix in sorted(working_formats.items()):
        print(f"    .{suffix:8s} -> {prefix}:")
    print(f"\n  These suffixes get added to _FINNHUB_EXCHANGE_MAP in gc_engine.py")
    print(f"  e.g. CIMB.KL -> {working_formats.get('KL','KLSE')}:CIMB")
else:
    print("  No working formats found — check API key and rate limits")

print(f"\n  Free tier: 60 calls/min, global coverage, no paywall on international.")
print(f"  Rate for our 527 empty tickers: ~10 min at safe pacing.")
