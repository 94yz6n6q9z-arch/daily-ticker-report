#!/usr/bin/env python3
"""
tools/sync_msci_manual_tickers.py
──────────────────────────────────────────────────────────────────────────────
Automatically keeps config/msci_manual_tickers.json up to date by fetching
live iShares ETF holdings and cross-referencing them against the countries
that have no dedicated ETF pipeline (Qatar, Kuwait, Saudi Arabia, New Zealand).

HOW IT WORKS
────────────
1. Fetch EIMI (Emerging Markets) holdings CSV
2. Filter rows where Country ∈ {Qatar, Kuwait, Saudi Arabia}
3. Run the same ticker-guessing logic used in update_msci_world_classification.py
4. Fetch IWDA (MSCI World) holdings CSV
5. Filter rows where Country == "New Zealand"
6. Diff each country's ticker set against the existing JSON
7. If any additions or removals found → rewrite JSON + exit 1 (signals workflow to commit)
8. If no changes → exit 0

EXIT CODES
──────────
0 = no changes detected
1 = JSON was updated (additions and/or removals found)
2 = fetch/parse error (non-fatal: workflow uses continue-on-error)

USAGE
─────
    python tools/sync_msci_manual_tickers.py [--json config/msci_manual_tickers.json]
    python tools/sync_msci_manual_tickers.py --dry-run   # print diff only, don't write

VERSION HISTORY
───────────────
1.0.0  Initial release. Monitors Qatar, Kuwait, Saudi Arabia, New Zealand.
       Fetches from EIMI (EM markets) and IWDA (World/NZL).
       Uses same parse/filter/guess logic as update_msci_world_classification.py.
"""

SYNC_VERSION = "1.0.0"

import argparse
import json
import sys
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── Import shared logic from sibling script ───────────────────────────────────
# We import the parsing/guessing functions from update_msci_world_classification
# to guarantee identical ticker normalization — no duplication, single source of truth.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from update_msci_world_classification import (
    SOURCE_CANDIDATES_EM,
    SOURCE_CANDIDATES_WORLD,
    COUNTRY_SUFFIX_FALLBACK,
    KNOWN_TICKER_OVERRIDES,
    SP500_11,
    fetch_holdings_csv,
    parse_ishares_holdings,
    filter_to_equities,
    guess_yahoo_ticker,
    canonical_sector,
    normalize_weight,
)

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_JSON = _REPO_ROOT / "config" / "msci_manual_tickers.json"

# Which countries to monitor, which source ETF to use, and the allow_numeric flag.
# "source" maps to one of the SOURCE_CANDIDATES_* lists.
MONITORED_COUNTRIES: List[Dict] = [
    {
        "country":       "Qatar",
        "source":        "em",   # Fetched from EIMI
        "allow_numeric": True,
        "suffix":        ".QA",
        "min_tickers":   5,      # Sanity floor — if fewer found, assume parse issue
        "notes":         "MSCI EM constituent since 2014. No standalone iShares Qatar ETF.",
    },
    {
        "country":       "Kuwait",
        "source":        "em",
        "allow_numeric": True,
        "suffix":        ".KW",
        "min_tickers":   3,
        "notes":         "Added to MSCI EM June 2020. Boursa Kuwait, Yahoo .KW suffix.",
    },
    {
        "country":       "Saudi Arabia",
        "source":        "em",
        "allow_numeric": True,
        "suffix":        ".SR",
        "min_tickers":   10,
        "notes":         "MSCI EM since 2019. Tadawul exchange, numeric tickers, Yahoo .SR suffix.",
    },
    {
        "country":       "New Zealand",
        "source":        "world",
        "allow_numeric": False,
        "suffix":        ".NZ",
        "min_tickers":   3,
        "notes":         "MSCI World. NZX-listed, Yahoo .NZ suffix. Typically 5 constituents.",
    },
]

# ── Fetch helpers ──────────────────────────────────────────────────────────────

_cache: Dict[str, object] = {}  # source_key -> (raw_df, source_as_of)


def _fetch_source(source_key: str, allow_numeric: bool):
    """Fetch and parse holdings for 'em' or 'world' source. Cached per run."""
    cache_key = f"{source_key}_{allow_numeric}"
    if cache_key in _cache:
        return _cache[cache_key]

    sources = SOURCE_CANDIDATES_EM if source_key == "em" else SOURCE_CANDIDATES_WORLD
    print(f"[sync] Fetching {source_key.upper()} holdings...", flush=True)
    try:
        fetched = fetch_holdings_csv(sources=sources)
        raw_df, source_as_of = parse_ishares_holdings(fetched.text)
        df = filter_to_equities(raw_df, allow_numeric=allow_numeric)
        _cache[cache_key] = (df, source_as_of, fetched.fund)
        print(f"[sync]   → {fetched.fund}: {len(df)} equity rows after filtering", flush=True)
    except Exception as e:
        print(f"[sync] ERROR fetching {source_key}: {e}", file=sys.stderr)
        raise
    return _cache[cache_key]


def _extract_country(source_key: str, country: str, allow_numeric: bool, suffix: str) -> List[Dict]:
    """Return list of {Ticker, Company, Sector, Country} dicts for one country."""
    df, _as_of, _fund = _fetch_source(source_key, allow_numeric)

    if "Country" not in df.columns:
        print(f"[sync] WARNING: no Country column in {source_key} data", file=sys.stderr)
        return []

    sub = df[df["Country"] == country].copy()
    if sub.empty:
        print(f"[sync] WARNING: no rows found for country='{country}' in {source_key}", file=sys.stderr)
        return []

    rows = []
    for _, row in sub.iterrows():
        raw_ticker = str(row.get("RawTicker", "")).strip()
        exchange   = str(row.get("Exchange", "")).strip()
        company    = str(row.get("Company", "")).strip()
        sector_raw = str(row.get("SectorRaw", row.get("Sector", ""))).strip()
        sector     = canonical_sector(sector_raw)

        ticker, _conf = guess_yahoo_ticker(raw_ticker, exchange)

        # Country-suffix fallback if Exchange-based guessing gives bare symbol
        if "." not in ticker and suffix:
            ticker = ticker + suffix

        # Apply known overrides
        ticker = KNOWN_TICKER_OVERRIDES.get(ticker, ticker)

        # Skip blank, multi-dot malformed, or non-SP500-sector tickers
        if not ticker or ticker.count(".") > 1:
            continue
        if sector not in SP500_11:
            continue

        rows.append({
            "Ticker":  ticker,
            "Company": company,
            "Country": country,
            "Sector":  sector,
        })

    # Deduplicate by ticker (keep first occurrence, highest weight usually first in CSV)
    seen = set()
    deduped = []
    for r in rows:
        if r["Ticker"] not in seen:
            seen.add(r["Ticker"])
            deduped.append(r)

    return deduped


# ── JSON schema ────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict:
    """Load existing JSON or return empty structure."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[sync] WARNING: could not parse existing JSON ({e}), starting fresh", file=sys.stderr)
    return {"_meta": {}, "countries": {}}


def _build_json(countries_data: Dict[str, List[Dict]], existing: Dict, fetch_ts: str) -> Dict:
    """Merge updated country data with existing metadata."""
    out = {
        "_meta": {
            "description": (
                "Auto-generated by tools/sync_msci_manual_tickers.py. "
                "Contains MSCI constituents for markets with no reliable iShares ETF data feed. "
                "Updated weekly by the refresh-msci-world GitHub Actions workflow. "
                "DO NOT edit manually — changes will be overwritten on the next sync run."
            ),
            "sync_version": SYNC_VERSION,
            "last_synced_utc": fetch_ts,
            "monitored_countries": [c["country"] for c in MONITORED_COUNTRIES],
        },
        "countries": {},
    }
    for cfg in MONITORED_COUNTRIES:
        country = cfg["country"]
        new_tickers = countries_data.get(country, [])
        prev_entry  = existing.get("countries", {}).get(country, {})
        prev_tickers = prev_entry.get("tickers", [])

        prev_set = {r["Ticker"] for r in prev_tickers}
        new_set  = {r["Ticker"] for r in new_tickers}
        added    = sorted(new_set - prev_set)
        removed  = sorted(prev_set - new_set)

        out["countries"][country] = {
            "notes":      cfg["notes"],
            "suffix":     cfg["suffix"],
            "source":     cfg["source"],
            "last_synced_utc": fetch_ts,
            "ticker_count": len(new_tickers),
            "added_since_last_sync":   added,
            "removed_since_last_sync": removed,
            "tickers": new_tickers,
        }
    return out


# ── Diff reporting ─────────────────────────────────────────────────────────────

def _report_diff(existing: Dict, updated: Dict) -> bool:
    """Print diff and return True if any changes were detected."""
    any_changes = False
    for country, entry in updated.get("countries", {}).items():
        added   = entry.get("added_since_last_sync", [])
        removed = entry.get("removed_since_last_sync", [])
        if added or removed:
            any_changes = True
            print(f"\n[sync] ⚡ {country}: {len(added)} added, {len(removed)} removed")
            if added:
                print(f"         + Added:   {', '.join(added)}")
            if removed:
                print(f"         - Removed: {', '.join(removed)}")
        else:
            n = entry.get("ticker_count", 0)
            print(f"[sync] ✓  {country}: {n} tickers — no change")
    return any_changes


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sync msci_manual_tickers.json against live EIMI/IWDA holdings"
    )
    ap.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help=f"Path to JSON output file (default: {DEFAULT_JSON})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diff only; do not write the JSON file",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Write JSON even if no changes detected",
    )
    args = ap.parse_args()

    json_path = Path(args.json)
    fetch_ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"[sync] sync_msci_manual_tickers.py v{SYNC_VERSION}")
    print(f"[sync] JSON target: {json_path}")
    print(f"[sync] Timestamp:   {fetch_ts}\n")

    existing = _load_json(json_path)

    # ── Fetch all countries ───────────────────────────────────────────────────
    countries_data: Dict[str, List[Dict]] = {}
    fetch_errors: List[str] = []

    for cfg in MONITORED_COUNTRIES:
        country    = cfg["country"]
        source_key = cfg["source"]
        allow_num  = cfg["allow_numeric"]
        suffix     = cfg["suffix"]
        min_t      = cfg["min_tickers"]
        print(f"[sync] Processing {country} (source: {source_key.upper()}, suffix: {suffix})")
        try:
            tickers = _extract_country(source_key, country, allow_num, suffix)
            if len(tickers) < min_t:
                print(
                    f"[sync] WARNING: only {len(tickers)} tickers found for {country} "
                    f"(expected ≥ {min_t}). Possible parse issue — keeping previous data.",
                    file=sys.stderr,
                )
                # Keep existing data for this country rather than wiping it
                prev = existing.get("countries", {}).get(country, {})
                countries_data[country] = prev.get("tickers", [])
            else:
                print(f"[sync]   → Found {len(tickers)} {suffix} tickers for {country}")
                countries_data[country] = tickers
        except Exception as e:
            print(f"[sync] ERROR processing {country}: {e}", file=sys.stderr)
            fetch_errors.append(country)
            # Keep existing data for failed country
            prev = existing.get("countries", {}).get(country, {})
            countries_data[country] = prev.get("tickers", [])

    # ── Build updated JSON and report diff ────────────────────────────────────
    updated = _build_json(countries_data, existing, fetch_ts)
    any_changes = _report_diff(existing, updated)

    if fetch_errors:
        print(f"\n[sync] WARNING: fetch errors for: {fetch_errors}. "
              f"Existing data kept for those countries.", file=sys.stderr)

    # ── Write if changed (or forced) ─────────────────────────────────────────
    if args.dry_run:
        print("\n[sync] --dry-run: JSON not written")
        return 1 if any_changes else 0

    if any_changes or args.force:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        action = "updated" if any_changes else "written (--force)"
        print(f"\n[sync] ✅ JSON {action}: {json_path}")
        # Exit 1 signals to the workflow that the file changed and needs committing
        return 1
    else:
        # Still write to update last_synced_utc and clear stale added/removed lists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n[sync] ✅ No constituent changes detected. Timestamps updated.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[sync] FATAL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
