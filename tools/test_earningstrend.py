"""
test_earningstrend.py — earningsTrend auth diagnostic & fix validator
=====================================================================
Runs against 30 tickers across all broken country groups.
Target: <10 min on GitHub Actions.

Tests 4 auth strategies in sequence per ticker, reports which one works
and what data it returns. This is the validation step before deploying
the fix into gc_engine.py.

Usage:
    pip install yfinance requests
    python test_earningstrend.py

Exit code 0 = at least one strategy works reliably.
Exit code 1 = all strategies failed (auth still broken).
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Test universe: 30 tickers covering every broken country + working controls
# ---------------------------------------------------------------------------
TEST_TICKERS = [
    # ── US controls (earnings_dates HTML — should always work via this method) ──
    ("AAPL",        "US",      "control"),
    ("MSFT",        "US",      "control"),
    ("NVDA",        "US",      "control"),
    # ── India .NS — was 0% in v0.6.6 (0/627 tickers) ──
    ("RELIANCE.NS", "India",   "broken"),
    ("TCS.NS",      "India",   "broken"),
    ("INFY.NS",     "India",   "broken"),
    ("HDFCBANK.NS", "India",   "broken"),
    ("WIPRO.NS",    "India",   "broken"),
    # ── South Korea .KS — was 0% (0/320) ──
    ("005930.KS",   "Korea",   "broken"),
    ("000660.KS",   "Korea",   "broken"),
    ("035420.KS",   "Korea",   "broken"),
    # ── Taiwan .TW — was 1% (4/319) ──
    ("2330.TW",     "Taiwan",  "broken"),
    ("2454.TW",     "Taiwan",  "broken"),
    ("2303.TW",     "Taiwan",  "broken"),
    # ── Japan .T — was 7% (12/179) ──
    ("7203.T",      "Japan",   "broken"),
    ("6758.T",      "Japan",   "broken"),
    ("9984.T",      "Japan",   "broken"),
    # ── Hong Kong .HK — was 2% (8/400) ──
    ("0700.HK",     "HongKong","broken"),
    ("0941.HK",     "HongKong","broken"),
    ("1299.HK",     "HongKong","broken"),
    # ── Europe (FMP working, earningsTrend should also work) ──
    ("ASML.AS",     "Europe",  "partial"),
    ("SAP.DE",      "Europe",  "partial"),
    ("LVMH.PA",     "Europe",  "partial"),
    # ── Canada .TO — was 62% (FMP covers, earningsTrend should add more) ──
    ("RY.TO",       "Canada",  "partial"),
    ("TD.TO",       "Canada",  "partial"),
    ("CNR.TO",      "Canada",  "partial"),
    # ── UK .L — was 36% ──
    ("SHEL.L",      "UK",      "partial"),
    ("AZN.L",       "UK",      "partial"),
    ("ULVR.L",      "UK",      "partial"),
]

BASE_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=earningsTrend%2CearningsHistory&corsDomain=finance.yahoo.com"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/quote/{ticker}/analysis",
}


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _parse_result(raw: Optional[Dict]) -> Optional[Dict]:
    """Return parsed data dict from raw quoteSummary response, or None."""
    if not raw:
        return None
    result = (raw.get("quoteSummary") or {}).get("result") or []
    if not result:
        return None
    return result[0]


def _summarise_data(data: Dict) -> Dict:
    """Extract coverage summary from parsed quoteSummary data block."""
    trend = data.get("earningsTrend", {}).get("trend", [])
    history = data.get("earningsHistory", {}).get("history", [])
    fwd_with_eps = sum(1 for t in trend if (t.get("earningsEstimate") or {}).get("avg") is not None)
    fwd_with_rev = sum(1 for t in trend if (t.get("revenueEstimate") or {}).get("avg") is not None)
    hist_with_eps = sum(1 for h in history if (h.get("epsEstimate") or {}).get("raw") is not None)
    return {
        "trend_rows": len(trend),
        "history_rows": len(history),
        "fwd_eps_rows": fwd_with_eps,
        "fwd_rev_rows": fwd_with_rev,
        "hist_eps_rows": hist_with_eps,
    }


def strategy_A(tk, ticker: str) -> Tuple[Optional[Dict], str, Optional[str]]:
    """
    Strategy A: tk._data.get_raw_json(url) WITHOUT handle_404.
    This is the v0.6.6 approach minus the bad kwarg.
    """
    url = BASE_URL.format(ticker=ticker)
    try:
        raw = tk._data.get_raw_json(url)
        data = _parse_result(raw)
        if data:
            return data, "A: get_raw_json (no handle_404)", None
        return None, "A: get_raw_json returned empty result", None
    except Exception as e:
        return None, "A: get_raw_json failed", str(e)


def strategy_B(tk, ticker: str) -> Tuple[Optional[Dict], str, Optional[str]]:
    """
    Strategy B: get crumb from yfinance session, append to URL, use tk._data.get_raw_json.
    """
    url = BASE_URL.format(ticker=ticker)
    try:
        crumb = None
        # Try common crumb accessors in yfinance
        for attr in ["get_crumb", "_get_crumb"]:
            if hasattr(tk._data, attr):
                try:
                    crumb = getattr(tk._data, attr)()
                    break
                except Exception:
                    pass
        # Also check if it's stored as a property
        if not crumb:
            for attr in ["crumb", "_crumb"]:
                c = getattr(tk._data, attr, None)
                if c and isinstance(c, str):
                    crumb = c
                    break

        if crumb:
            url = url + f"&crumb={crumb}"

        raw = tk._data.get_raw_json(url)
        data = _parse_result(raw)
        if data:
            return data, f"B: get_raw_json+crumb (crumb={'yes' if crumb else 'no'})", None
        return None, f"B: empty result (crumb={'yes' if crumb else 'no'})", None
    except Exception as e:
        return None, "B: get_raw_json+crumb failed", str(e)


def strategy_C(tk, ticker: str) -> Tuple[Optional[Dict], str, Optional[str]]:
    """
    Strategy C: Use the curl_cffi session that yfinance 1.2.0 uses internally.
    curl_cffi handles TLS fingerprinting and cookies. Access via tk._data.session
    or by getting the session from the ticker's data object.
    """
    url = BASE_URL.format(ticker=ticker)
    try:
        # yfinance 1.2.0 uses curl_cffi — the session is available via multiple paths
        sess = None
        for path in [
            lambda: tk._data.session,
            lambda: tk._data.cache,
            lambda: getattr(tk, "_session", None),
        ]:
            try:
                sess = path()
                if sess is not None:
                    break
            except Exception:
                pass

        if sess is None:
            return None, "C: no session found", None

        # Try to get crumb and append
        crumb = None
        try:
            if hasattr(sess, "crumb"):
                crumb = sess.crumb
        except Exception:
            pass

        req_url = url + (f"&crumb={crumb}" if crumb else "")
        resp = sess.get(req_url, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()
            data = _parse_result(raw)
            if data:
                return data, f"C: curl_cffi session (crumb={'yes' if crumb else 'no'})", None
            return None, "C: session returned empty result", None
        return None, f"C: session HTTP {resp.status_code}", None
    except Exception as e:
        return None, "C: session failed", str(e)


def strategy_D(ticker: str) -> Tuple[Optional[Dict], str, Optional[str]]:
    """
    Strategy D: Plain requests.get with browser headers (old gc_engine-5.py approach).
    Worked in older version — tests whether GitHub Actions IPs are blocked.
    Note: uses full ticker symbol (e.g. RELIANCE.NS), not stripped.
    """
    url = BASE_URL.format(ticker=ticker)
    headers = {**BROWSER_HEADERS, "Referer": BROWSER_HEADERS["Referer"].format(ticker=ticker)}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()
            data = _parse_result(raw)
            if data:
                return data, "D: plain requests+browser_headers", None
            return None, "D: plain requests returned empty result", None
        # Check if it's an auth error
        try:
            err = resp.json().get("quoteSummary", {}).get("error", {})
        except Exception:
            err = {}
        return None, f"D: HTTP {resp.status_code} err={err.get('code','?')}", None
    except Exception as e:
        return None, "D: plain requests failed", str(e)


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 70)
    print(f"earningsTrend Auth Strategy Test  —  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"yfinance version: {yf.__version__}")
    print("=" * 70)

    results = []
    strategy_wins = {"A": 0, "B": 0, "C": 0, "D": 0, "none": 0}

    for ticker, country, group in TEST_TICKERS:
        print(f"\n{'─'*60}")
        print(f"  {ticker:15s} [{country:10s}] ({group})")
        print(f"{'─'*60}")

        # Initialise yfinance Ticker (triggers crumb/cookie fetch)
        tk = yf.Ticker(ticker)

        # Warm up — pull info to ensure session is authenticated
        try:
            _ = tk.fast_info
        except Exception:
            pass

        winner = None
        winner_label = "none"
        summary = None

        for strat_fn, label in [
            (lambda: strategy_A(tk, ticker),     "A"),
            (lambda: strategy_B(tk, ticker),     "B"),
            (lambda: strategy_C(tk, ticker),     "C"),
            (lambda: strategy_D(ticker),          "D"),
        ]:
            data, desc, err = strat_fn()
            status = "✅ PASS" if data else "❌ FAIL"
            print(f"  Strategy {label}: {status}  {desc}")
            if err:
                print(f"             err={err[:100]}")
            if data and winner is None:
                winner = data
                winner_label = label
                summary = _summarise_data(data)
                print(f"             → trend_rows={summary['trend_rows']}  "
                      f"fwd_eps={summary['fwd_eps_rows']}  "
                      f"fwd_rev={summary['fwd_rev_rows']}  "
                      f"hist_eps={summary['hist_eps_rows']}")

        strategy_wins[winner_label] += 1
        results.append({
            "ticker": ticker,
            "country": country,
            "group": group,
            "winner": winner_label,
            "summary": summary,
        })
        time.sleep(0.3)  # polite pacing

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Coverage by country group
    by_group = {}
    for r in results:
        g = r["country"]
        by_group.setdefault(g, {"total": 0, "hit": 0, "strategy": []})
        by_group[g]["total"] += 1
        if r["winner"] != "none":
            by_group[g]["hit"] += 1
            by_group[g]["strategy"].append(r["winner"])

    print(f"\n{'Country':<12} {'Tickers':>8} {'Got data':>10} {'Coverage':>10} {'Strategy wins'}")
    print("-" * 60)
    for country, s in sorted(by_group.items()):
        strats = ", ".join(sorted(set(s["strategy"]))) if s["strategy"] else "none"
        pct = s["hit"] / s["total"] * 100
        flag = "✅" if pct == 100 else ("⚠️" if pct > 0 else "❌")
        print(f"{country:<12} {s['total']:>8} {s['hit']:>10} {pct:>9.0f}%  {flag}  {strats}")

    print()
    print(f"Strategy win counts across all {len(TEST_TICKERS)} tickers:")
    for strat, wins in strategy_wins.items():
        print(f"  Strategy {strat}: {wins} tickers")

    broken_group = [r for r in results if r["group"] == "broken"]
    broken_hits = sum(1 for r in broken_group if r["winner"] != "none")
    print()
    print(f"CRITICAL COUNTRIES (broken group): {broken_hits}/{len(broken_group)} tickers got earningsTrend data")

    # Best strategy recommendation
    best_strat = max(["A","B","C","D"], key=lambda s: strategy_wins[s])
    best_wins = strategy_wins[best_strat]
    total = len(TEST_TICKERS)

    print()
    if best_wins == total:
        print(f"✅ RECOMMENDATION: Strategy {best_strat} works for all {total} tickers — use this in gc_engine.")
    elif best_wins >= total * 0.8:
        print(f"⚠️  RECOMMENDATION: Strategy {best_strat} works for {best_wins}/{total} tickers.")
        print(f"   Combine with next-best strategy for full coverage.")
    elif broken_hits == 0:
        print("❌ NO STRATEGY WORKED for broken countries. Auth still blocked on this environment.")
        print("   Check if GitHub Actions IPs are being geo-blocked by Yahoo.")
    else:
        print(f"⚠️  Partial success. Best strategy: {best_strat} ({best_wins}/{total}).")

    # Dump JSON for CI artifact
    output = {
        "run_date": datetime.utcnow().isoformat(),
        "yfinance_version": yf.__version__,
        "strategy_wins": strategy_wins,
        "broken_hits": broken_hits,
        "broken_total": len(broken_group),
        "results": results,
    }
    with open("earningstrend_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print()
    print("Full results saved to: earningstrend_test_results.json")

    exit_code = 0 if broken_hits > 0 else 1
    return exit_code


if __name__ == "__main__":
    ec = run_tests()
    sys.exit(ec)
