#!/usr/bin/env python3
"""
FMP Coverage Pre-Check
Answers "is the paid plan worth it?" using only FREE tier endpoints.
"""

import json, os, sys, time, urllib.parse, urllib.request

FMP_BASE = "https://financialmodelingprep.com/stable"
API_KEY  = os.environ.get("FMP_API_KEY", "").strip()
if not API_KEY:
    print("FMP_API_KEY not set."); sys.exit(1)

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

# Test 1: Key validity
print("\n-- Test 1: Key validity --")
data, err = fmp_get("/income-statement", {"symbol": "AAPL", "period": "quarter", "limit": 1})
if err:
    print(f"[FAIL] Key error: {err}"); sys.exit(1)
if data and isinstance(data, list) and len(data) > 0:
    print(f"[OK] Key valid - free tier working")
else:
    print(f"[WARN] Unexpected: {str(data)[:100]}")

# Test 2: Symbol coverage using free /search-symbol
print("\n-- Test 2: Coverage check for empty tickers (free endpoint) --")

EMPTY_TICKERS = {
    "KL":  ["ABMB.KL","AEONCR.KL","AFFIN.KL","AMBANK.KL","AXREIT.KL","BIMB.KL","BURSA.KL",
            "CARLSBG.KL","CIMB.KL","DIALOG.KL","DIGI.KL","GENTING.KL","GENM.KL","HLFG.KL",
            "HLBANK.KL","IHH.KL","IOICORP.KL","KLCC.KL","KLK.KL","MAYBANK.KL"],
    "IS":  ["AKBNK.IS","ARCLK.IS","BIMAS.IS","DOHOL.IS","EKGYO.IS","ENKAI.IS","EREGL.IS",
            "FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS","KCHOL.IS","KOZAL.IS","PGSUS.IS",
            "SAHOL.IS","SISE.IS","TCELL.IS","THYAO.IS","TOASO.IS","TTKOM.IS"],
    "PS":  ["BDO.PS","BPI.PS","JGS.PS","MBT.PS","MER.PS","SM.PS","SMPH.PS","TEL.PS","URC.PS"],
    "AD":  ["FAB.AD","ADNOCDIST.AD","ALDAR.AD","ALPHADHABI.AD","ETISALAT.AD","TAQA.AD"],
    "DU":  ["DIB.DU","DEWA.DU","EMAAR.DU","DFM.DU","ENBD.DU"],
    "US_ghost": ["ABCB10","ABCB4","ABEV3","ALPHA","AFLT","AGUAS-A","ALOS3","ALPA4","ARZAN"],
}
FULL_COUNTS = {"KL":81,"IS":74,"PS":31,"AD":20,"DU":21,"US_ghost":226}

results = {}
for exch, tickers in EMPTY_TICKERS.items():
    found, not_found = 0, []
    print(f"\n  .{exch} (sampling {len(tickers)} of {FULL_COUNTS.get(exch,len(tickers))}):")
    for t in tickers:
        base = t.rsplit(".",1)[0] if "." in t else t
        data, _ = fmp_get("/search-symbol", {"query": base, "limit": 5})
        time.sleep(0.2)
        matched = False
        if data and isinstance(data, list):
            for item in data:
                sym = item.get("symbol","")
                if sym == t or sym == base or sym.startswith(base+"."):
                    matched = True; break
        if matched: found += 1
        else: not_found.append(t)
    pct = found/len(tickers)*100
    status = "OK  " if pct>=70 else ("WARN" if pct>=30 else "FAIL")
    print(f"    [{status}] {found}/{len(tickers)} found ({pct:.0f}%)")
    if not_found:
        print(f"       Not in FMP: {', '.join(not_found[:5])}" + (f" +{len(not_found)-5} more" if len(not_found)>5 else ""))
    results[exch] = {"found":found,"total":len(tickers),"pct":pct}

# Summary
print("\n-- Summary --")
print(f"{'Exchange':12s} {'Full count':>10} {'Est. recover':>13}  {'Hit%':>6}  Verdict")
print("-"*55)
total = 0
for exch, res in results.items():
    full = FULL_COUNTS.get(exch, res["total"])
    est  = int(full * res["pct"] / 100)
    total += est
    v = "WORTH IT" if res["pct"]>=70 else ("PARTIAL" if res["pct"]>=30 else "SKIP")
    print(f".{exch:11s} {full:>10}  ~{est:>11}  {res['pct']:>5.0f}%  {v}")

print(f"\n  Estimated recoverable with paid plan: ~{total} tickers")
print(f"  Current empty: 527  |  Total universe: 4,167")
print(f"\n  NOTE: symbol hit rate != earnings data available.")
print(f"  Real recovery likely 10-20% lower. Paid plan starts ~$15/mo.")
