#!/usr/bin/env python3
"""
add_slugs.py — Daily tool for manually adding investing.com slugs to gc_engine.py.

Targets only tickers that ALREADY have yfinance EPS data (the ~1,654 coverage universe)
but are missing revenue estimates. Ordered by region: US -> EU/UK -> KR -> JP -> RoW.

Usage:
    python3 add_slugs.py --stats                         # coverage overview
    python3 add_slugs.py --todo                          # today's 100 tickers
    python3 add_slugs.py --todo --day 3                  # day 3 batch
    python3 add_slugs.py --todo --region us              # US only
    python3 add_slugs.py --todo --all                    # full list
    python3 add_slugs.py --add NVDA=nvidia-corp BA=boeing-co
    python3 add_slugs.py --verify NVDA BA GD
"""

import argparse, json, re, sys, time
from pathlib import Path

STATE_FILE  = "gc_state.json"
ENGINE_FILE = "gc_engine.py"
BATCH_SIZE  = 100

REGION_ORDER = ["US", "EU_UK", "KR", "JP", "RoW"]

REGION_SUFFIXES = {
    "US":    set(),
    "EU_UK": {"L","DE","PA","AS","MI","MC","SW","ST","OL","HE","CO","BR","VI","IR","LS","AT"},
    "KR":    {"KS","KQ"},
    "JP":    {"T"},
    "RoW":   None,
}

def _region(ticker):
    suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else ""
    if not suffix:
        return "US"
    for region, suffixes in REGION_SUFFIXES.items():
        if region == "RoW":
            continue
        if suffixes and suffix in suffixes:
            return region
    return "RoW"

def _region_rank(ticker):
    return REGION_ORDER.index(_region(ticker))

def load_state():
    if not Path(STATE_FILE).exists():
        print(f"ERROR: {STATE_FILE} not found. Run from repo root.")
        sys.exit(1)
    with open(STATE_FILE) as f:
        raw = json.load(f)
    state = raw.get("earnings_cache", raw)
    first_val = next(iter(state.values()), None)
    if not isinstance(first_val, dict):
        for val in raw.values():
            if isinstance(val, dict):
                inner = next(iter(val.values()), None)
                if isinstance(inner, dict):
                    return val
    return state

def load_current_slugs():
    src = Path(ENGINE_FILE).read_text()
    slugs = {}
    in_block = False
    for line in src.splitlines():
        if "SLUG_OVERRIDES" in line and "Dict" in line:
            in_block = True
            continue
        if in_block:
            if line.strip() == "}":
                break
            m = re.match(r'\s+"([A-Z0-9.^-]+)":\s+"([^"]+)"', line)
            if m:
                slugs[m.group(1)] = m.group(2)
    return slugs

def get_targets(state, current_slugs, region_filter=None):
    """Tickers with analyst EPS estimates but no IC slug yet.
    We target tickers with eps_estimate (analyst coverage), not just eps_reported.
    We do NOT skip tickers that already have revenue_estimate — FMP may have filled
    those temporarily, but we still need a permanent IC slug for when FMP expires."""
    candidates = []
    for ticker, data in state.items():
        if data.get("inactive"):
            continue
        # Target universe: tickers where analysts publish EPS estimates
        has_eps_est = any(r.get("eps_estimate") is not None
                          for r in data.get("earnings_dates", []))
        if not has_eps_est:
            continue
        # Skip only if already has a confirmed IC slug — not based on rev est presence
        bare = ticker.split(".")[0].upper()
        if bare in current_slugs:
            continue
        region = _region(ticker)
        if region_filter and region.upper() != region_filter.upper():
            continue
        eps_qs = len([r for r in data.get("earnings_dates", [])
                      if r.get("eps_reported") is not None])
        candidates.append((_region_rank(ticker), -eps_qs, ticker, region))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [(r, t) for _, _, t, r in candidates]

def add_slugs_to_engine(new_slugs):
    src = Path(ENGINE_FILE).read_text()
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
        print("ERROR: Could not find SLUG_OVERRIDES in gc_engine.py")
        return 0
    new_lines = [f'        "{t}":  "{s}",' for t, s in sorted(new_slugs.items())]
    lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
    Path(ENGINE_FILE).write_text("\n".join(lines) + "\n")
    return len(new_lines)

def verify_slugs(tickers):
    try:
        from gc_engine import enrich_estimates_investing_com
    except ImportError:
        print("ERROR: gc_engine.py not found.")
        sys.exit(1)
    import datetime
    def stub_ed():
        rows = []
        today = datetime.date.today()
        for q in range(8):
            months_back = q * 3 + 1
            y = today.year; m = today.month - months_back
            while m <= 0: m += 12; y -= 1
            rows.append({"date": f"{y:04d}-{m:02d}-15", "eps_estimate": None,
                         "eps_reported": 1.0, "eps_surprise_pct": None,
                         "revenue_estimate": None, "revenue_reported": None})
        return rows
    print(f"\nVerifying {len(tickers)} tickers...\n")
    print(f"{'':2}{'Ticker':<12}  {'RevEst':>10}  {'RevAct':>10}  {'Qs':>8}  {'Time':>5}")
    print("-" * 56)
    for ticker in tickers:
        ed = stub_ed()
        t0 = time.time()
        enrich_estimates_investing_com(ed, ticker)
        elapsed = time.time() - t0
        ests = [r["revenue_estimate"] for r in ed if r.get("revenue_estimate") is not None]
        acts = [r["revenue_reported"]  for r in ed if r.get("revenue_reported")  is not None]
        def fmt(v):
            if v is None: return "–"
            if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:.0f}"
        ok = "✓" if ests else "✗"
        print(f"{ok} {ticker:<12}  {fmt(ests[0] if ests else None):>10}  "
              f"{fmt(acts[0] if acts else None):>10}  "
              f"{len(ests):>3}e/{len(acts):<3}a  {elapsed:>4.1f}s")
        time.sleep(2.0)

def show_stats(state, current_slugs):
    inactive   = sum(1 for d in state.values() if d.get("inactive"))
    active     = len(state) - inactive
    yf_covered = sum(1 for d in state.values()
                     if not d.get("inactive")
                     and any(r.get("eps_estimate") is not None
                             for r in d.get("earnings_dates", [])))
    has_rev    = sum(1 for t, d in state.items()
                     if not d.get("inactive")
                     and any(r.get("revenue_estimate") is not None
                             for r in d.get("earnings_dates", [])
                             if r.get("eps_estimate") is not None))
    targets    = get_targets(state, current_slugs)
    by_region  = {}
    for region, ticker in targets:
        by_region.setdefault(region, 0)
        by_region[region] += 1

    print(f"\n{'='*55}")
    print(f"Coverage overview\n")
    print(f"  Total active tickers:          {active}")
    print(f"  Inactive/dead:                 {inactive}")
    print(f"  yfinance EPS covered:          {yf_covered}  ← target universe")
    print(f"  Have revenue estimates:        {has_rev}")
    print(f"  Still need slugs:              {len(targets)}")
    print(f"  Days to complete @ {BATCH_SIZE}/day:    ~{len(targets)//BATCH_SIZE + 1}")
    print(f"\n  By region:")
    for r in REGION_ORDER:
        n = by_region.get(r, 0)
        if n:
            days = n // BATCH_SIZE + (1 if n % BATCH_SIZE else 0)
            print(f"    {r:<8}  {n:>4} tickers  (~{days} day{'s' if days!=1 else ''})")
    print(f"{'='*55}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo",   action="store_true")
    ap.add_argument("--all",    action="store_true")
    ap.add_argument("--day",    type=int, default=0)
    ap.add_argument("--region", type=str, default="")
    ap.add_argument("--add",    nargs="+", metavar="TICKER=slug")
    ap.add_argument("--verify", nargs="+", metavar="TICKER")
    ap.add_argument("--stats",  action="store_true")
    args = ap.parse_args()

    if not any([args.todo, args.add, args.verify, args.stats]):
        ap.print_help()
        return

    state = load_state()
    slugs = load_current_slugs()

    if args.stats:
        show_stats(state, slugs)
        return

    if args.todo:
        targets = get_targets(state, slugs, region_filter=args.region or None)
        batch   = [t for _, t in targets] if args.all else \
                  [t for _, t in targets[args.day*BATCH_SIZE:(args.day+1)*BATCH_SIZE]]
        rlabel  = f" [{args.region.upper()}]" if args.region else ""
        dlabel  = "All" if args.all else f"Day {args.day+1}"
        print(f"\n{dlabel}{rlabel} — {len(batch)} tickers  ({len(targets)} total remaining)\n")
        print("For each ticker:")
        print("  1. Go to https://www.investing.com/search/?q=TICKER")
        print("  2. Click the stock → go to Earnings tab")
        print("  3. Copy slug from URL: .../equities/SLUG-earnings")
        print(f"  4. python3 add_slugs.py --add TICKER=slug\n")
        print(f"{'#':>4}  {'Ticker':<16}  {'Region':<8}  {'EPS Qs':>6}  Search link")
        print("-" * 76)
        prev_region = None
        for i, ticker in enumerate(batch, 1):
            d      = state.get(ticker, {})
            region = _region(ticker)
            eps_qs = len([r for r in d.get("earnings_dates", [])
                          if r.get("eps_reported") is not None])
            bare   = ticker.split(".")[0].lower()
            if region != prev_region and prev_region is not None:
                print()
            prev_region = region
            n = args.day * BATCH_SIZE + i
            print(f"{n:>4}  {ticker:<16}  {region:<8}  {eps_qs:>6}  "
                  f"https://www.investing.com/search/?q={bare}")
        print(f"\nWhen done: python3 add_slugs.py --verify {' '.join(batch[:5])}")
        return

    if args.add:
        new_slugs = {}
        for item in args.add:
            if "=" not in item:
                print(f"ERROR: bad format '{item}' — use TICKER=slug"); continue
            ticker, slug = item.split("=", 1)
            if "investing.com/equities/" in slug:
                slug = slug.split("/equities/")[1].rstrip("/").replace("-earnings","")
            new_slugs[ticker.upper()] = slug.strip()
        if not new_slugs:
            return
        dupes = {t for t in new_slugs if t in slugs}
        if dupes:
            print(f"Updating: {dupes}")
        n = add_slugs_to_engine(new_slugs)
        print(f"\n✓ {n} slug(s) added to gc_engine.py:")
        for t, s in sorted(new_slugs.items()):
            print(f"  {t:<14} → {s}")
        print(f"\nNext: python3 add_slugs.py --verify {' '.join(new_slugs.keys())}")
        return

    if args.verify:
        verify_slugs(args.verify)

if __name__ == "__main__":
    main()
