#!/usr/bin/env python3
"""
gc_state_cleanup.py
===================
One-shot maintenance script: removes stale entries from gc_state.json
that are no longer in the MSCI universe (World + EM CSVs).

After the switch from 9-ETF universe (~6,400 tickers) to 2-ETF universe
(~2,100 tickers), approximately 4,200 tickers remain in gc_state.json
from the old EIMI / country-ETF sources. They will never be fetched
again by gc_engine (since load_universe() no longer returns them) but
they waste disk space and make gc_state harder to inspect.

Additionally, tickers listed in tickers_custom.txt are always preserved
regardless of MSCI membership.

Usage
-----
    # Dry run (shows what would be removed — safe):
    python tools/gc_state_cleanup.py

    # Apply (actually deletes the stale entries):
    python tools/gc_state_cleanup.py --apply

Output
------
  gc_state_backup_<date>.json  — backup created before any modification
  gc_state.json                — updated in-place (only with --apply)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
DOCS_DIR       = BASE_DIR / "docs"
CONFIG_DIR     = BASE_DIR / "config"
GC_STATE_PATH  = DOCS_DIR / "gc_state.json"
CUSTOM_TICKERS = BASE_DIR / "tickers_custom.txt"
WORLD_CSV      = CONFIG_DIR / "msci_world_classification.csv"
EM_CSV         = CONFIG_DIR / "msci_em_classification.csv"


def load_universe_tickers() -> set[str]:
    """Load all tickers from both MSCI CSVs."""
    import csv
    tickers: set[str] = set()
    for csv_path in (WORLD_CSV, EM_CSV):
        if not csv_path.exists():
            print(f"[cleanup] WARNING: {csv_path} not found — universe may be incomplete")
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("Ticker", "").strip()
                if t:
                    tickers.add(t)
    return tickers


def load_custom_tickers() -> set[str]:
    """Load tickers_custom.txt (always preserved in gc_state)."""
    if not CUSTOM_TICKERS.exists():
        return set()
    lines = CUSTOM_TICKERS.read_text(encoding="utf-8").splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove stale tickers from gc_state.json")
    parser.add_argument("--apply", action="store_true",
                        help="Actually modify gc_state.json (default: dry-run only)")
    args = parser.parse_args()

    if not GC_STATE_PATH.exists():
        print(f"[cleanup] gc_state.json not found at {GC_STATE_PATH}")
        return

    print(f"[cleanup] loading gc_state.json …")
    with open(GC_STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)

    all_tickers_in_state = set(state.keys())
    print(f"[cleanup] gc_state has {len(all_tickers_in_state)} entries")

    universe = load_universe_tickers()
    print(f"[cleanup] MSCI universe: {len(universe)} tickers")

    custom = load_custom_tickers()
    print(f"[cleanup] custom tickers: {len(custom)} (always preserved)")

    keep = universe | custom
    stale = all_tickers_in_state - keep

    print(f"\n[cleanup] STALE (not in MSCI + not in custom): {len(stale)}")
    print(f"[cleanup] KEEP  (in MSCI or custom):           {len(all_tickers_in_state - stale)}")

    if not stale:
        print("[cleanup] Nothing to remove — gc_state is already clean.")
        return

    # Show a sample of stale tickers grouped by suffix
    from collections import Counter
    suffix_counts = Counter(
        t.rsplit(".", 1)[-1] if "." in t else "US"
        for t in stale
    )
    print("\n[cleanup] Stale tickers by exchange suffix (top 20):")
    for suffix, count in suffix_counts.most_common(20):
        print(f"  .{suffix:6s}  {count}")

    if not args.apply:
        print(f"\n[cleanup] DRY RUN — no changes made.")
        print(f"[cleanup] Run with --apply to remove {len(stale)} stale entries.")
        return

    # Backup
    backup_path = DOCS_DIR / f"gc_state_backup_{date.today().isoformat()}.json"
    shutil.copy2(GC_STATE_PATH, backup_path)
    print(f"\n[cleanup] Backup saved → {backup_path}")

    # Remove stale
    for t in stale:
        del state[t]

    with open(GC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[cleanup] Removed {len(stale)} stale entries from gc_state.json")
    print(f"[cleanup] gc_state now has {len(state)} entries")


if __name__ == "__main__":
    main()
