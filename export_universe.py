#!/usr/bin/env python3
"""
export_universe.py  —  Universe review export  v5
==================================================
Produces universe_review.csv: every ticker in the MSCI World + EM universe
with four anchor columns:
  1. msci_raw_ticker  — original symbol as printed in the iShares XLS
  2. yahoo_ticker     — normalized Yahoo Finance symbol (after TICKER_OVERRIDES
                        and _normalize_ticker rules — identical to gc_engine input)
  3. fmp_ticker       — best FMP symbol for this ticker (ADR or bare symbol)
  4. company_name     — company name as printed in the iShares XLS

Deduplication is on yahoo_ticker — if the same normalized symbol appears in
multiple source CSVs, it produces ONE row. The msci_raw_ticker column shows
the first-seen raw symbol (world CSV has priority over EM).

Mcap columns:
  mcap_usd_b      — USD market cap from _mcap_usd in gc_state (FX-converted)
  mcap_local_b    — local-currency market cap from info.market_cap
  mcap_currency   — currency of mcap_local_b (e.g. HKD, INR, CHF)

Fully standalone — no universe.py / gc_engine.py imports required.
Keep _TICKER_OVERRIDES, _ADR_MAP, and _normalize_ticker in sync with universe.py.

VERSION HISTORY
v1-v3  Initial versions, exchange/country fixes, _last_known_mcap fallback
v4     _normalize_ticker applied in _read_csv_tickers() — fixes phantom no_cache entries
v5     Full column restructure: msci_raw_ticker/yahoo_ticker/fmp_ticker/company_name anchor
       columns; dedup on yahoo_ticker eliminates duplicate rows; mcap split into
       mcap_usd_b + mcap_local_b + mcap_currency; floor updated to $1B unified;
       Malaysia .KL no longer labelled dead.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR   = BASE_DIR / "docs"

DEFAULT_STATE = DOCS_DIR / "gc_state.json"
DEFAULT_OUT   = BASE_DIR / "universe_review.csv"

EXPORT_VERSION = "5.0.0"

MIN_MCAP_US_EU = 1_000_000_000
MIN_MCAP_OTHER = 1_000_000_000

EU_SUFFIXES = {
    "L", "DE", "PA", "AS", "MI", "MC", "ST", "OL", "HE", "CO",
    "LS", "BR", "AT", "IR", "SW", "WA", "VI",
}

_TICKER_OVERRIDES = {
    "2299955D.TO": "CSU.TO",
    "BRK.B":       "BRK-B",
    "BRKB":        "BRK-B",
    "HEI.A":       "HEI-A",
    "HEIA":        "HEI-A",
    "HEI.B":       "HEI",
    "TITIM.MI":    "TIT.MI",
    "CICT.SI":     "CPAMF",
    "BAAKOMB.PR":  "KOMB.PR",
    "532483.BO":   "CANBK.BO",
    "SHFL.NS":     "SHRIRAMFIN.BO",
}

_ADR_MAP = {
    "2330.TW": "TSM",   "2303.TW": "UMC",   "3711.TW": "ASX",
    "7203.T":  "TM",    "6758.T":  "SONY",  "7974.T":  "NTDOY",
    "9432.T":  "NTTYY", "9984.T":  "SFTBY", "9433.T":  "KDDIY",
    "4519.T":  "CHGCY", "6954.T":  "FANUY", "8001.T":  "ITOCY",
    "4063.T":  "SHECY", "6501.T":  "HTHIY", "8035.T":  "TOELY",
    "6857.T":  "AVANF", "0700.HK": "TCEHY", "9988.HK": "BABA",
    "9618.HK": "JD",    "9999.HK": "NTES",  "2454.TW": "MDTKF",
    "3690.HK": "MPNGF", "2382.HK": "SMPRY", "6367.T":  "DAIIF",
}


def _normalize_ticker(t):
    t = t.strip()
    if not t:
        return t
    if t in _TICKER_OVERRIDES:
        return _TICKER_OVERRIDES[t]
    t = re.sub(r'\.\.[A-Z]', lambda m: '.' + m.group(0)[-1], t)
    t = re.sub(r'\.\.([A-Z]+)$', r'.\1', t)
    t = re.sub(r'\.E\.IS$', '.IS', t)
    t = re.sub(r'-E\.IS$',   '.IS', t)
    t = re.sub(r'\.R\.BK$', '.BK', t)
    t = re.sub(r'\.(AD|DU)$', '.AE', t)
    t = re.sub(r'^([A-Z][A-Z0-9]*)\.([A-Z][A-Z0-9]*)\.([A-Z]+)$', r'\1-\2.\3', t)
    t = re.sub(r'\*\.', '.', t)
    return t


def _get_fmp_ticker(yahoo_ticker):
    if yahoo_ticker in _ADR_MAP:
        return _ADR_MAP[yahoo_ticker]
    return yahoo_ticker.split(".")[0].upper()


def _exch(ticker):
    return ticker.rsplit(".", 1)[-1] if "." in ticker else "US"


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


EXCHANGE_COUNTRY = {
    "US": "United States",  "TO": "Canada",        "SA": "Brazil",
    "MX": "Mexico",         "SN": "Chile",         "CL": "Colombia",
    "CA": "Egypt",          "L":  "United Kingdom","DE": "Germany",
    "PA": "France",         "AS": "Netherlands",   "MI": "Italy",
    "MC": "Spain",          "ST": "Sweden",        "OL": "Norway",
    "HE": "Finland",        "CO": "Denmark",       "LS": "Portugal",
    "BR": "Belgium",        "IR": "Ireland",       "SW": "Switzerland",
    "WA": "Poland",         "VI": "Austria",       "AT": "Greece",
    "PR": "Czech Republic", "BD": "Hungary",
    "SR": "Saudi Arabia",   "QA": "Qatar",         "KW": "Kuwait",
    "AD": "UAE (Abu Dhabi)","DU": "UAE (Dubai)",   "AE": "UAE",
    "JO": "South Africa",   "TA": "Israel",
    "T":  "Japan",          "TW": "Taiwan",        "HK": "Hong Kong",
    "TWO":"Taiwan (OTC)",   "SS": "China (Shanghai)","SZ":"China (Shenzhen)",
    "KS": "South Korea",    "KQ": "South Korea",   "NS": "India (NSE)",
    "BO": "India (BSE)",    "SI": "Singapore",     "AX": "Australia",
    "NZ": "New Zealand",    "KL": "Malaysia",      "PS": "Philippines",
    "JK": "Indonesia",      "BK": "Thailand",      "IS": "Turkey",
}

CSV_DEFS = [
    ("world", "msci_world_classification.csv"),
    ("em",    "msci_em_classification.csv"),
]

FIELDNAMES = [
    "msci_raw_ticker", "yahoo_ticker", "fmp_ticker", "company_name",
    "country", "exchange", "source_csvs", "in_em_csv",
    "mcap_usd_b", "mcap_local_b", "mcap_currency",
    "mcap_source", "mcap_floor_b", "passes_mcap",
    "status", "no_data_runs", "ghost_risk",
    "has_eps", "has_revenue", "data_gap",
    "inactive_reason", "inactive_since", "fetched_at", "error",
]


def _read_csv_rows(path):
    if not path.exists():
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw   = (row.get("RawTicker") or "").strip()
            yahoo = (row.get("Ticker") or row.get("ticker") or "").strip()
            if not yahoo:
                continue
            yahoo_norm = _normalize_ticker(yahoo)
            result.append({
                "raw_ticker":   raw or yahoo,
                "yahoo_ticker": yahoo_norm,
                "company":      (row.get("Company") or row.get("company") or "").strip(),
                "country_csv":  (row.get("Country") or "").strip(),
                "exchange_csv": (row.get("Exchange") or "").strip(),
            })
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
        print(f"[export] Loading {state_path} ...")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cache = state.get("earnings_cache", {})
        print(f"[export] gc_state cache entries: {len(cache):,}")
    else:
        print(f"[export] WARNING: {state_path} not found")

    ticker_meta    = {}
    ticker_sources = defaultdict(set)
    csv_counts     = {}

    for source_key, csv_fname in CSV_DEFS:
        path = CONFIG_DIR / csv_fname
        rows = _read_csv_rows(path)
        csv_counts[source_key] = len(rows)
        for r in rows:
            yt = r["yahoo_ticker"]
            if not yt:
                continue
            ticker_sources[yt].add(source_key)
            if yt not in ticker_meta:
                ticker_meta[yt] = {
                    "raw_ticker":   r["raw_ticker"],
                    "company":      r["company"],
                    "country_csv":  r["country_csv"],
                    "exchange_csv": r["exchange_csv"],
                }

    for yt in cache:
        if yt not in ticker_sources:
            ticker_sources[yt].add("state_only")
            if yt not in ticker_meta:
                ticker_meta[yt] = {
                    "raw_ticker":   yt,
                    "company":      cache[yt].get("info", {}).get("short_name", ""),
                    "country_csv":  "",
                    "exchange_csv": "",
                }

    all_tickers = sorted(ticker_sources.keys())

    print(f"\n[export] Source CSV counts (after normalization):")
    for source_key, _ in CSV_DEFS:
        n = csv_counts.get(source_key, 0)
        print(f"  {source_key:8s}: {n:5,}")
    print(f"  After dedup on yahoo_ticker: {len(all_tickers):,} unique tickers")
    overlap = sum(1 for s in ticker_sources.values() if len(s) > 1)
    print(f"  Multi-source:  {overlap:,} tickers in 2+ CSVs (one row each)")

    risk_1 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 1)
    risk_2 = sum(1 for v in cache.values() if v.get("_no_data_runs") == 2)
    print(f"\n[export] Ghost risk: MED={risk_1:,}  HIGH={risk_2:,}")

    rows = []
    for yahoo_ticker in all_tickers:
        meta    = ticker_meta.get(yahoo_ticker, {})
        entry   = cache.get(yahoo_ticker, {})
        info    = entry.get("info") or {}
        sources = ticker_sources[yahoo_ticker]

        msci_raw  = meta.get("raw_ticker", yahoo_ticker)
        company   = meta.get("company", info.get("short_name", ""))
        fmp_tick  = _get_fmp_ticker(yahoo_ticker)

        ex      = _exch(yahoo_ticker)
        country = EXCHANGE_COUNTRY.get(ex, f"Unknown ({ex})")

        mcap_usd   = entry.get("_mcap_usd")
        mcap_local = info.get("market_cap")
        mcap_curr  = info.get("currency", "")

        mcap_source = ""
        if entry.get("_mcap_source"):
            mcap_source = entry["_mcap_source"]
        elif mcap_usd and mcap_local:
            mcap_source = "yf"
        elif entry.get("_last_known_mcap"):
            mcap_source = "cached"

        if not mcap_usd and entry.get("_last_known_mcap"):
            mcap_usd = entry["_last_known_mcap"]
            mcap_source = "cached_local"

        floor_raw   = _mcap_floor(yahoo_ticker)
        floor_b_str = f"{floor_raw / 1e9:.0f}"

        try:
            mcap_usd_f = float(mcap_usd) if mcap_usd else 0.0
        except (TypeError, ValueError):
            mcap_usd_f = 0.0

        passes_mcap = "?" if mcap_usd_f <= 0 else ("Y" if mcap_usd_f >= floor_raw else "N")

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
            "msci_raw_ticker": msci_raw,
            "yahoo_ticker":    yahoo_ticker,
            "fmp_ticker":      fmp_tick,
            "company_name":    company,
            "country":         country,
            "exchange":        ex,
            "source_csvs":     ",".join(sorted(sources)),
            "in_em_csv":       "Y" if "em" in sources else "",
            "mcap_usd_b":      _fmt_b(mcap_usd),
            "mcap_local_b":    _fmt_b(mcap_local),
            "mcap_currency":   mcap_curr,
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

    rows.sort(key=lambda r: (r["country"], r["yahoo_ticker"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[export] Written {len(rows):,} rows -> {out_path}")

    sc = Counter(r["status"] for r in rows)
    print(f"\n[export] Status breakdown:")
    for s, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {n:5,}")

    mc_yes = sum(1 for r in rows if r["passes_mcap"] == "Y")
    mc_no  = sum(1 for r in rows if r["passes_mcap"] == "N")
    mc_unk = sum(1 for r in rows if r["passes_mcap"] == "?")
    print(f"\n[export] Mcap filter (${floor_b_str}B floor):")
    print(f"  Passes: {mc_yes:,}  Below: {mc_no:,}  No data: {mc_unk:,}")

    print(f"\n[export] Active tickers by country (top 25):")
    ca = Counter(r["country"] for r in rows if r["status"] == "active")
    for country, n in ca.most_common(25):
        print(f"  {country:35s}: {n:4,}")

    dup_check = Counter(r["yahoo_ticker"] for r in rows)
    dups = {t: c for t, c in dup_check.items() if c > 1}
    if dups:
        print(f"\n[export] WARNING: {len(dups)} duplicate yahoo_tickers:")
        for t, c in sorted(dups.items()):
            print(f"  {t}: {c} rows")
    else:
        print(f"\n[export] Dedup check: OK - no duplicate yahoo_tickers")

    print(f"\n[export] Done.")


if __name__ == "__main__":
    main()
