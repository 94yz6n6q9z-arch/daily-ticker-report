#!/usr/bin/env python3
"""
add_slugs.py — Daily tool for manually adding investing.com slugs to gc_engine.py.

WORKFLOW:
  1. Run with --todo to see today's batch of 50 tickers that need slugs
  2. For each ticker, go to investing.com, search for it, copy the earnings page URL
  3. Run with --add to patch gc_engine.py with the new slugs
  4. Run with --verify to confirm they all return data

Usage:
    python3 add_slugs.py --todo                          # show today's 50 missing slugs
    python3 add_slugs.py --todo --all                    # show ALL missing slugs
    python3 add_slugs.py --add NVDA=nvidia-corp          # add one slug
    python3 add_slugs.py --add NVDA=nvidia-corp AAPL=apple-computer-inc  # add many
    python3 add_slugs.py --verify NVDA AAPL MSFT         # test slugs return data
    python3 add_slugs.py --stats                         # show coverage stats from state
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

STATE_FILE   = "gc_state.json"
ENGINE_FILE  = "gc_engine.py"
BATCH_SIZE   = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        print(f"ERROR: {STATE_FILE} not found. Run from repo root.")
        sys.exit(1)
    with open(STATE_FILE) as f:
        return json.load(f)


def load_current_slugs() -> dict:
    """Extract the SLUG_OVERRIDES dict from gc_engine.py."""
    src = Path(ENGINE_FILE).read_text()
    # Find all "TICKER": "slug" pairs inside SLUG_OVERRIDES
    slugs = {}
    in_block = False
    for line in src.splitlines():
        if "SLUG_OVERRIDES" in line and "Dict" in line:
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("}"):
                break
            m = re.match(r'\s+"([A-Z0-9.^]+)":\s+"([^"]+)"', line)
            if m:
                slugs[m.group(1)] = m.group(2)
    return slugs


def get_missing_tickers(state: dict, current_slugs: dict) -> list:
    """Return tickers that have no revenue estimates and no slug yet, ordered by market cap proxy."""
    missing = []
    for ticker, data in state.items():
        if data.get("inactive"):
            continue
        # Check if this ticker has revenue estimates already
        has_rev_est = any(
            r.get("revenue_estimate") is not None
            for r in data.get("earnings_dates", [])
            if r.get("eps_reported") is not None
        )
        if has_rev_est:
            continue
        # Skip if already has a slug
        bare = ticker.split(".")[0].upper()
        if bare in current_slugs:
            continue
        # Only include tickers that have some data (not complete dead)
        has_eps = any(r.get("eps_reported") is not None for r in data.get("earnings_dates", []))
        has_rev = len(data.get("quarterly_revenue", [])) > 0
        if not has_eps and not has_rev:
            continue
        # Score by data richness (proxy for importance)
        score = len(data.get("earnings_dates", [])) + len(data.get("quarterly_revenue", [])) * 2
        missing.append((score, ticker))

    # Sort by score descending (most data-rich = most likely to have IC page)
    missing.sort(key=lambda x: -x[0])
    return [t for _, t in missing]


def add_slugs_to_engine(new_slugs: dict) -> int:
    """Patch SLUG_OVERRIDES in gc_engine.py with new ticker→slug mappings."""
    src = Path(ENGINE_FILE).read_text()

    # Find insertion point: just before the closing } of SLUG_OVERRIDES
    # We look for the last entry line before the closing brace
    lines = src.splitlines()
    insert_idx = None
    in_block = False
    for i, line in enumerate(lines):
        if "SLUG_OVERRIDES" in line and "Dict" in line:
            in_block = True
        if in_block and line.strip() == "}":
            insert_idx = i
            break

    if insert_idx is None:
        print("ERROR: Could not find SLUG_OVERRIDES closing brace in gc_engine.py")
        return 0

    # Build new lines
    new_lines = []
    for ticker, slug in sorted(new_slugs.items()):
        new_lines.append(f'        "{ticker}":  "{slug}",')

    # Insert before closing brace
    lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
    Path(ENGINE_FILE).write_text("\n".join(lines) + "\n")
    return len(new_lines)


def verify_slugs(tickers: list) -> None:
    """Test that the given tickers now return data from investing.com."""
    try:
        from gc_engine import enrich_estimates_investing_com
    except ImportError:
        print("ERROR: gc_engine.py not found.")
        sys.exit(1)

    import datetime

    def stub_ed(n=8):
        rows = []
        today = datetime.date.today()
        for q in range(n):
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
            })
        return rows

    print(f"\nVerifying {len(tickers)} tickers...\n")
    print(f"{'Ticker':<8}  {'RevEst Q1':>12}  {'RevAct Q1':>12}  {'Quarters':>8}  {'Time':>5}")
    print("-" * 55)

    for ticker in tickers:
        ed = stub_ed()
        t0 = time.time()
        filled = enrich_estimates_investing_com(ed, ticker)
        elapsed = time.time() - t0
        rev_ests = [r["revenue_estimate"] for r in ed if r.get("revenue_estimate") is not None]
        rev_acts = [r["revenue_reported"] for r in ed if r.get("revenue_reported") is not None]

        def fmt(v):
            if v is None: return "–"
            if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:.0f}"

        ok = "✓" if rev_ests else "✗"
        print(f"{ok} {ticker:<6}  {fmt(rev_ests[0] if rev_ests else None):>12}  "
              f"{fmt(rev_acts[0] if rev_acts else None):>12}  "
              f"{len(rev_ests):>3}est/{len(rev_acts):<3}act  {elapsed:>4.1f}s")
        time.sleep(2.0)


def show_stats(state: dict, current_slugs: dict) -> None:
    """Show current coverage stats."""
    total = len([t for t, d in state.items() if not d.get("inactive")])
    has_rev_est = sum(
        1 for t, d in state.items()
        if not d.get("inactive")
        and any(r.get("revenue_estimate") is not None
                for r in d.get("earnings_dates", [])
                if r.get("eps_reported") is not None)
    )
    has_slug = sum(
        1 for t in state
        if not state[t].get("inactive")
        and t.split(".")[0].upper() in current_slugs
    )
    inactive = sum(1 for d in state.values() if d.get("inactive"))
    missing = get_missing_tickers(state, current_slugs)

    print(f"\n{'='*55}")
    print(f"Coverage stats from {STATE_FILE}\n")
    print(f"  Total active tickers:        {total}")
    print(f"  Inactive/dead (skipped):     {inactive}")
    print(f"  Have revenue estimates:      {has_rev_est} ({has_rev_est*100//max(total,1)}%)")
    print(f"  Have IC slug:                {has_slug} ({has_slug*100//max(total,1)}%)")
    print(f"  Missing slug + no rev est:   {len(missing)}")
    print(f"  Days to complete at 50/day:  ~{len(missing)//50 + 1}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Manage investing.com slugs for gc_engine.py")
    ap.add_argument("--todo",   action="store_true", help="Show today's batch of missing slugs")
    ap.add_argument("--all",    action="store_true", help="Show ALL missing (not just today's batch)")
    ap.add_argument("--add",    nargs="+", metavar="TICKER=slug", help="Add slug(s): NVDA=nvidia-corp")
    ap.add_argument("--verify", nargs="+", metavar="TICKER", help="Verify slugs return IC data")
    ap.add_argument("--stats",  action="store_true", help="Show coverage stats")
    ap.add_argument("--day",    type=int, default=0, help="Which day's batch to show (0=today, 1=tomorrow...)")
    args = ap.parse_args()

    if not any([args.todo, args.add, args.verify, args.stats]):
        ap.print_help()
        return

    state = load_state()
    current_slugs = load_current_slugs()

    if args.stats:
        show_stats(state, current_slugs)
        return

    if args.todo:
        missing = get_missing_tickers(state, current_slugs)
        total = len(missing)
        if args.all:
            batch = missing
        else:
            start = args.day * BATCH_SIZE
            batch = missing[start: start + BATCH_SIZE]

        day_label = f"Day {args.day + 1}" if not args.all else "All"
        print(f"\n{day_label} — {len(batch)} tickers to look up "
              f"({'of ' + str(total) + ' total' if not args.all else 'total'})\n")
        print("For each ticker below:")
        print("  1. Go to https://www.investing.com")
        print("  2. Search for the ticker, click through to Earnings page")
        print("  3. Copy the URL slug (the part after /equities/ and before -earnings)")
        print("  4. Run: python3 add_slugs.py --add TICKER=slug\n")
        print(f"{'#':>3}  {'Ticker':<10}  {'EPS Qs':>6}  {'Rev Qs':>6}  investing.com URL to find")
        print("-" * 70)
        for i, ticker in enumerate(batch, 1):
            d = state.get(ticker, {})
            eps_qs = len([r for r in d.get("earnings_dates", []) if r.get("eps_reported") is not None])
            rev_qs = len(d.get("quarterly_revenue", []))
            bare = ticker.split(".")[0].lower()
            print(f"{i:>3}  {ticker:<10}  {eps_qs:>6}  {rev_qs:>6}  "
                  f"https://www.investing.com/search/?q={bare}")
        print(f"\nWhen done, run: python3 add_slugs.py --verify {' '.join(batch[:5])}")
        return

    if args.add:
        new_slugs = {}
        for item in args.add:
            if "=" not in item:
                print(f"ERROR: bad format '{item}' — use TICKER=slug")
                continue
            ticker, slug = item.split("=", 1)
            # Extract slug from full URL if user pasted the whole thing
            if "investing.com/equities/" in slug:
                slug = slug.split("/equities/")[1].rstrip("/").replace("-earnings", "")
            new_slugs[ticker.upper()] = slug.strip()

        if not new_slugs:
            return

        # Check for duplicates
        existing = load_current_slugs()
        dupes = {t: s for t, s in new_slugs.items() if t in existing}
        if dupes:
            print(f"Already in slug table (will update): {dupes}")

        added = add_slugs_to_engine(new_slugs)
        print(f"\n✓ Added {added} slug(s) to gc_engine.py:")
        for t, s in new_slugs.items():
            print(f"  {t} → {s}")
        print(f"\nNext: python3 add_slugs.py --verify {' '.join(new_slugs.keys())}")
        return

    if args.verify:
        verify_slugs(args.verify)
        return


if __name__ == "__main__":
    main()
