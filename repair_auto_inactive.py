#!/usr/bin/env python3
"""
repair_auto_inactive.py
=======================
One-shot repair: clears tickers incorrectly auto-inactived by the three
degraded yfinance runs on 2026-03-11 and 2026-03-12.

Root cause: yfinance crumb expired (4-worker parallel bug + 2w-exp failure),
causing 4,500+ tickers to return empty data on every run. The persistent
_no_data_runs counter hit >=3 and marked 3,306 tickers as inactive even
though they are real, live tickers with data.

Safe to run: only clears tickers marked inactive_reason='persistent_no_data_3_runs'
AND inactive_since >= 2026-03-11. Tickers genuinely inactived before the
degradation event are untouched.

Usage:
    python repair_auto_inactive.py [--dry-run] [--state path/to/gc_state.json]
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_STATE = Path(__file__).parent / "docs" / "gc_state.json"
CUTOFF_DATE  = "2026-03-11"   # Only clear tickers inactived on/after this date


def load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path: Path, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, separators=(",", ":"))
    print(f"[repair] State saved → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing anything")
    ap.add_argument("--state", default=str(DEFAULT_STATE),
                    help="Path to gc_state.json")
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"[repair] ERROR: {state_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"[repair] Loading {state_path} …")
    state = load(state_path)
    cache = state.get("earnings_cache", {})

    if not cache:
        print("[repair] No earnings_cache found — nothing to do")
        sys.exit(0)

    # ── Identify candidates ──────────────────────────────────────────────────
    cleared   = []
    skipped   = []

    for ticker, data in cache.items():
        reason = data.get("inactive_reason", "")
        since  = data.get("inactive_since", "")

        if reason != "persistent_no_data_3_runs":
            continue
        if since < CUTOFF_DATE:
            # Inactived before the degradation event — leave alone
            skipped.append((ticker, since, reason))
            continue

        cleared.append(ticker)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n[repair] Summary:")
    print(f"  Total cache entries          : {len(cache):,}")
    print(f"  Degraded-run inactive (clear): {len(cleared):,}")
    print(f"  Pre-degradation inactive (keep): {len(skipped):,}")

    if not cleared:
        print("[repair] Nothing to clear — already clean.")
        sys.exit(0)

    print(f"\n[repair] First 20 tickers to be cleared:")
    for t in cleared[:20]:
        d = cache[t]
        print(f"  {t:15s}  inactive_since={d.get('inactive_since','?')[:10]}  "
              f"_no_data_runs={d.get('_no_data_runs','?')}")
    if len(cleared) > 20:
        print(f"  … and {len(cleared)-20} more")

    if args.dry_run:
        print("\n[repair] DRY RUN — no changes written.")
        return

    # ── Apply fixes ──────────────────────────────────────────────────────────
    for ticker in cleared:
        d = cache[ticker]
        d.pop("inactive",            None)
        d.pop("inactive_since",      None)
        d.pop("inactive_reason",     None)
        d.pop("_no_data_runs",       None)
        d.pop("_used_cached",        None)
        d.pop("_cache_fallback_reason",   None)
        d.pop("_cache_fallback_run_date", None)
        # Keep any real data that may be present, just un-flag as inactive
        # Force re-fetch next run by clearing fetched_at so _should_fetch_today returns True
        d.pop("fetched_at", None)

    print(f"\n[repair] Cleared {len(cleared):,} tickers — writing state …")
    save(state_path, state)
    print(f"[repair] Done. Run gc_engine --mode data (without --force) to rebuild.")


if __name__ == "__main__":
    main()
