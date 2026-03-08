#!/usr/bin/env python3
"""
FMP API diagnostic — run locally with:
    FMP_API_KEY=your_key python3 test_fmp.py

Tests:
  1. Key validity
  2. Symbol conversion (Yahoo → FMP format) for each problem exchange
  3. Actual data return for 1 ticker per exchange
"""

import json
import os
import sys
import urllib.parse
import urllib.request

FMP_BASE = "https://financialmodelingprep.com/api"

API_KEY = os.environ.get("FMP_API_KEY", "").strip()
if not API_KEY:
    print("✗ FMP_API_KEY not set. Run: FMP_API_KEY=your_key python3 test_fmp.py")
    sys.exit(1)

# ── Colour helpers ────────────────────────────────────────────────
OK   = "✓"
FAIL = "✗"
WARN = "⚠"

def fmp_get(path, params=None):
    params = {**(params or {}), "apikey": API_KEY}
    url = f"{FMP_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
            if isinstance(data, dict) and data.get("Error Message"):
                return None, data["Error Message"]
            return data, None
    except Exception as e:
        return None, str(e)


# ── Test 1: Key validity ──────────────────────────────────────────
print("\n── Test 1: Key validity ─────────────────────────────────────")
data, err = fmp_get("/v3/profile/AAPL")
if err:
    print(f"{FAIL} Key invalid or network error: {err}")
    sys.exit(1)
if data and isinstance(data, list) and data[0].get("symbol") == "AAPL":
    print(f"{OK} Key valid — AAPL profile returned OK")
else:
    print(f"{WARN} Unexpected response: {str(data)[:100]}")


# ── Test 2: Symbol mapping ────────────────────────────────────────
# Yahoo Finance suffix  →  what we send to FMP  →  correct?
# Current _FMP_SUFFIX_MAP strips unknown suffixes (returns base only)
# Problem: .KL .IS .PS .AD .DU are NOT in the map → suffix gets stripped
print("\n── Test 2: Symbol mapping for problem exchanges ─────────────")

CURRENT_SUFFIX_MAP = {
    "L": None, "KS": "KS", "T": "T", "HK": "HK", "TW": "TW",
    "PA": "PA", "DE": "DE", "SW": "SW", "AS": "AS", "MI": "MI",
    "MC": "MC", "ST": "ST", "OL": "OL", "HE": "HE", "CO": "CO",
    "TO": "TO", "AX": "AX", "NS": "NS", "SA": "SA",
}

test_cases = [
    ("CIMB.KL",    "KL",  "CIMB.KL",   "Malaysia — FMP uses .KL"),
    ("EREGL.IS",   "IS",  "EREGL.IS",  "Turkey — FMP uses .IS"),
    ("BDO.PS",     "PS",  "BDO.PS",    "Philippines — FMP uses .PS"),
    ("FAB.AD",     "AD",  "FAB.AD",    "Abu Dhabi — FMP uses .AD"),
    ("DIB.DU",     "DU",  "DIB.DU",    "Dubai — FMP uses .DU"),
    ("7203.T",     "T",   "7203.T",    "Japan — already in map"),
    ("AZN.L",      "L",   "AZN",       "UK LSE — strip suffix (correct)"),
    ("CBA.AX",     "AX",  "CBA.AX",   "Australia — already in map"),
]

mapping_issues = []
for yahoo, suffix, expected_fmp, note in test_cases:
    base = yahoo.rsplit(".", 1)[0]
    fmp_suffix = CURRENT_SUFFIX_MAP.get(suffix)  # None if not in map
    if fmp_suffix is None and suffix in CURRENT_SUFFIX_MAP:
        actual = base          # explicitly mapped to None = strip
    elif fmp_suffix is None:
        actual = base          # NOT in map → silently strips suffix ← BUG
    else:
        actual = f"{base}.{fmp_suffix}"

    ok = actual == expected_fmp
    status = OK if ok else FAIL
    if not ok:
        mapping_issues.append((yahoo, actual, expected_fmp))
    print(f"  {status} {yahoo:12s} → sent as '{actual:12s}'  (expected '{expected_fmp}')  {note}")

if mapping_issues:
    print(f"\n{FAIL} {len(mapping_issues)} mapping issues — FMP receives wrong symbols for these exchanges")
else:
    print(f"\n{OK} All mappings correct")


# ── Test 3: Live data fetch for 1 ticker per exchange ─────────────
print("\n── Test 3: Live data return per exchange ────────────────────")

# Use correct FMP symbols (what they SHOULD be after fix)
test_tickers = [
    ("CIMB.KL",  "CIMB.KL",  "Malaysia .KL"),
    ("EREGL.IS", "EREGL.IS", "Turkey .IS"),
    ("BDO.PS",   "BDO.PS",   "Philippines .PS"),
    ("FAB.AD",   "FAB.AD",   "Abu Dhabi .AD"),
    ("DIB.DU",   "DIB.DU",   "Dubai .DU"),
    ("7203.T",   "7203.T",   "Japan .T"),
    ("AZN.L",    "AZN",      "UK .L (stripped)"),
    ("CBA.AX",   "CBA.AX",   "Australia .AX"),
    ("AAPL",     "AAPL",     "US control"),
]

for yahoo, fmp_sym, label in test_tickers:
    rev_data, err1 = fmp_get(f"/v3/income-statement/{fmp_sym}", {"period": "quarter", "limit": 4})
    eps_data, err2 = fmp_get(f"/v3/earnings-surprises/{fmp_sym}")

    has_rev = isinstance(rev_data, list) and len(rev_data) > 0
    has_eps = isinstance(eps_data, list) and len(eps_data) > 0

    if has_rev and has_eps:
        latest_rev = rev_data[0].get("revenue", "?")
        status = OK
        detail = f"rev={latest_rev:,.0f}  eps_quarters={len(eps_data)}"
    elif has_rev:
        status = WARN
        detail = f"revenue OK, no EPS surprises"
    elif has_eps:
        status = WARN
        detail = f"EPS OK, no revenue"
    else:
        status = FAIL
        detail = f"no data  (rev_err={err1}  eps_err={err2})"

    print(f"  {status} {yahoo:12s} (FMP: {fmp_sym:12s})  {label:20s}  {detail}")

# ── Summary ───────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────")
if mapping_issues:
    print(f"{FAIL} MAPPING BUG: These suffixes are stripped before reaching FMP:")
    for yahoo, actual, expected in mapping_issues:
        print(f"     {yahoo} → sent as '{actual}' but should be '{expected}'")
    print(f"\n  Fix: add these to _FMP_SUFFIX_MAP in gc_engine.py:")
    for _, _, expected in mapping_issues:
        sfx = expected.rsplit(".", 1)[-1] if "." in expected else None
        if sfx:
            print(f"     \"{sfx}\": \"{sfx}\",")
else:
    print(f"{OK} No mapping issues found")
    print(f"{OK} FMP key is active and returning data")
    print(f"   Next nightly run should recover .KL, .IS, .PS, .AD, .DU tickers")
