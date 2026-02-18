#!/usr/bin/env python3
"""
Daily Technical Triggers Report
Universe:
- Nasdaq-100 (Wikipedia)
- S&P 500 (Wikipedia)
- Custom watchlist: config/tickers_custom.txt  (supports international suffix tickers)

Patterns:
- H&S Top / Inverse H&S (screening-level)
- Triangles: Sym / Asc / Desc (trendline convergence with flat-side detection)
- Wedges (converging non-triangle)
- Broadening formations (diverging trendlines)
- Rectangles (range + low drift)

Outputs:
- docs/report.md + docs/index.md
- docs/state.json (previous signals → enables NEW/ENDED diffs)
- docs/img/YYYY-MM-DD/*.png (annotated charts only for tickers with signals)

Run behavior:
- Scheduled: around 22:30 Europe/Berlin (config below)
- Manual testing: set FORCE_RUN=1 (recommended in workflow)
"""

import os
import re
import json
import math
from datetime import datetime, timedelta

import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -------------------- Settings --------------------
TZ = "Europe/Berlin"
RUN_AT_HOUR = 22
RUN_AT_MIN = 30

LOOKBACK = "3y"
CHUNK = 25

PIVOT_W = 6
WINDOW = 260                 # ~1 trading year for patterns
NEAR_PCT = 3.0               # watch threshold
BREAK_PCT = 1.0              # trigger threshold beyond boundary (close-based)

MAX_CHARTS_PER_UNIVERSE = 25 # cap charts per universe (NDX/SPX) to keep repo light
CUSTOM_AFTERHOURS_ONLY = True

STATE_PATH = "docs/state.json"
REPORT_MD = "docs/report.md"
INDEX_MD = "docs/index.md"


# -------------------- Time gating --------------------
def now_local():
    return datetime.now(pytz.timezone(TZ))

def should_run_now():
    # For manual runs in Actions: set env FORCE_RUN=1
    if os.getenv("FORCE_RUN") == "1":
        return True
    t = now_local()
    return (t.hour == RUN_AT_HOUR) and (abs(t.minute - RUN_AT_MIN) <= 7)


# -------------------- Ticker normalization --------------------
def normalize_ticker(sym: str) -> str:
    """
    Keep exchange suffix tickers intact: MUV2.DE, 000660.KS, MC.PA, UCG.MI, RMS.PA
    Convert only US class-share tickers like BRK.B or BF.B -> BRK-B / BF-B
    """
    sym = str(sym).strip().upper()
    if re.match(r"^[A-Z]{1,6}\.[A-Z]$", sym):
        return sym.replace(".", "-")
    return sym


# -------------------- Ticker universes --------------------
def get_nasdaq100_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        if "Ticker" in t.columns:
            return [normalize_ticker(x) for x in t["Ticker"].astype(str).tolist()]
    return []

def get_sp500_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for t in tables:
        if "Symbol" in t.columns:
            return [normalize_ticker(x) for x in t["Symbol"].astype(str).tolist()]
    return []

def read_custom_tickers(path="config/tickers_custom.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        ticks = [x.strip() for x in f.readlines()]
    ticks = [x for x in ticks if x and not x.startswith("#")]
    return [normalize_ticker(x) for x in ticks]


# -------------------- Data download --------------------
def download_chunked(tickers):
    parts = []
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i+CHUNK]
        dfp = yf.download(
            part,
            period=LOOKBACK,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        parts.append(dfp)
    return pd.concat(parts, axis=1)


# -------------------- Helpers --------------------
def pct(a, b):
    return (a - b) / b * 100.0 if b not in (0, None) else np.nan

def pivots(series: pd.Series, w: int = PIVOT_W):
    roll_max = series.rolling(2*w+1, center=True).max()
    roll_min = series.rolling(2*w+1, center=True).min()
    piv_hi = (series == roll_max).fillna(False)
    piv_lo = (series == roll_min).fillna(False)
    return series[piv_hi], series[piv_lo]

def fit_line(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return None
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)

def line_y(a, b, x):
    return a*x + b


# -------------------- After-hours snapshot (custom only) --------------------
def after_hours_snapshot(ticker: str):
    """
    Compute after-hours move from regular close to latest pre/post 1m bar (Yahoo).
    Often best for US tickers; EU/KR tickers may be empty (normal).
    Returns dict: reg_close, last_px, ah_change_pct (NaN if unavailable)
    """
    out = {"reg_close": np.nan, "last_px": np.nan, "ah_change_pct": np.nan}
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="2d", interval="1m", prepost=True)
        if h is None or h.empty:
            return out
        h = h.dropna()
        out["last_px"] = float(h["Close"].iloc[-1])

        idx = h.index
        # Heuristic: bar around 16:00 local exchange time
        mask = (idx.hour == 16) & (idx.minute <= 5)
        if mask.any():
            out["reg_close"] = float(h.loc[mask, "Close"].iloc[-1])
        else:
            prev_day = (idx[-1].date() - timedelta(days=1))
            prev = h[h.index.date == prev_day]
            if not prev.empty:
                out["reg_close"] = float(prev["Close"].iloc[-1])

        if not math.isnan(out["reg_close"]) and not math.isnan(out["last_px"]):
            out["ah_change_pct"] = pct(out["last_px"], out["reg_close"])
        return out
    except Exception:
        return out


# -------------------- Pattern detection --------------------
def detect_rectangle(raw: pd.DataFrame, n=WINDOW):
    w = raw.tail(n)
    if len(w) < 80:
        return None
    hi = w["High"].quantile(0.95)
    lo = w["Low"].quantile(0.05)
    rng = (hi - lo) / max(1e-9, lo)
    drift = pct(w["Close"].iloc[-1], w["Close"].iloc[0])
    if rng < 0.25 and abs(drift) < 20:
        return {"support": float(lo), "resistance": float(hi)}
    return None

def detect_trendlines(raw: pd.DataFrame, n=WINDOW):
    """
    Fit trendlines to last 4 pivot highs and lows (Close).
    Determine converge/diverge and classify triangle/wedge/broadening.
    """
    w = raw.tail(n).copy().reset_index(drop=True)
    close = w["Close"]

    hi, lo = pivots(close, w=PIVOT_W)
    hi_idx = np.array(hi.index.to_list(), dtype=int)
    lo_idx = np.array(lo.index.to_list(), dtype=int)

    if len(hi_idx) < 3 or len(lo_idx) < 3:
        return None

    hi_idx = hi_idx[-4:]
    lo_idx = lo_idx[-4:]

    upper = fit_line(hi_idx, close.iloc[hi_idx].to_numpy())
    lower = fit_line(lo_idx, close.iloc[lo_idx].to_numpy())
    if upper is None or lower is None:
        return None

    aU, bU = upper
    aL, bL = lower

    last = len(w) - 1
    early = max(0, last - 120)

    gap_now = line_y(aU, bU, last) - line_y(aL, bL, last)
    gap_then = line_y(aU, bU, early) - line_y(aL, bL, early)

    flatU = abs(aU) < 0.02
    flatL = abs(aL) < 0.02
    diverge = gap_then > 0 and (gap_now > gap_then * 1.15)
    converge = gap_then > 0 and (gap_now < gap_then * 0.85)

    if diverge:
        patt = "Broadening"
    elif converge:
        if (aU < 0 and aL > 0):
            patt = "Sym Triangle"
        elif flatU and aL > 0:
            patt = "Ascending Triangle"
        elif flatL and aU < 0:
            patt = "Descending Triangle"
        else:
            patt = "Wedge"
    else:
        patt = "Channel/Other"

    return {
        "pattern": patt,
        "aU": float(aU), "bU": float(bU),
        "aL": float(aL), "bL": float(bL),
        "gap_now": float(gap_now), "gap_then": float(gap_then),
    }

def boundary_now(raw: pd.DataFrame, tl: dict):
    w = raw.tail(WINDOW).copy().reset_index(drop=True)
    last = len(w) - 1
    upper = line_y(tl["aU"], tl["bU"], last)
    lower = line_y(tl["aL"], tl["bL"], last)
    return float(upper), float(lower)

def detect_hs_top(raw: pd.DataFrame, n=WINDOW):
    """
    Screening H&S Top:
    - pivot highs define LS/Head/RS
    - neckline from lowest lows between LS-Head and Head-RS (linear)
    """
    w = raw.tail(n).copy().reset_index(drop=True)
    if len(w) < 140:
        return None

    close = w["Close"]
    low = w["Low"]

    hi, _ = pivots(close, w=PIVOT_W)
    idx = hi.index.to_list()
    if len(idx) < 6:
        return None

    best = None
    for i in range(len(idx) - 2):
        ls, hd, rs = idx[i], idx[i+1], idx[i+2]
        span = rs - ls
        if span < 25 or span > 220:
            continue

        ls_p, hd_p, rs_p = float(close.iloc[ls]), float(close.iloc[hd]), float(close.iloc[rs])
        if hd_p < max(ls_p, rs_p) * 1.03:
            continue
        if abs(ls_p - rs_p) / ((ls_p + rs_p) / 2) > 0.15:
            continue

        t1 = int(low.iloc[ls:hd+1].idxmin())
        t2 = int(low.iloc[hd:rs+1].idxmin())
        t1p, t2p = float(low.loc[t1]), float(low.loc[t2])

        last = len(w) - 1
        neck = t1p + (t2p - t1p) * ((last - t1) / max(1, (t2 - t1)))
        last_close = float(close.iloc[last])

        best = {
            "neck": float(neck),
            "last_close": float(last_close),
        }
    return best

def detect_inv_hs(raw: pd.DataFrame, n=WINDOW):
    """
    Screening Inverse H&S:
    - pivot lows define LS/Head/RS
    - neckline from highest highs between LS-Head and Head-RS (linear)
    """
    w = raw.tail(n).copy().reset_index(drop=True)
    if len(w) < 140:
        return None

    close = w["Close"]
    high = w["High"]

    _, lo = pivots(close, w=PIVOT_W)
    idx = lo.index.to_list()
    if len(idx) < 6:
        return None

    best = None
    for i in range(len(idx) - 2):
        ls, hd, rs = idx[i], idx[i+1], idx[i+2]
        span = rs - ls
        if span < 25 or span > 260:
            continue

        ls_p, hd_p, rs_p = float(close.iloc[ls]), float(close.iloc[hd]), float(close.iloc[rs])
        if hd_p > min(ls_p, rs_p) * 0.97:
            continue
        if abs(ls_p - rs_p) / ((ls_p + rs_p) / 2) > 0.22:
            continue

        p1 = int(high.iloc[ls:hd+1].idxmax())
        p2 = int(high.iloc[hd:rs+1].idxmax())
        p1p, p2p = float(high.loc[p1]), float(high.loc[p2])

        last = len(w) - 1
        neck = p1p + (p2p - p1p) * ((last - p1) / max(1, (p2 - p1)))
        last_close = float(close.iloc[last])

        best = {
            "neck": float(neck),
            "last_close": float(last_close),
        }
    return best


# -------------------- Signals + overlays --------------------
def compute_signals_for_ticker(raw: pd.DataFrame):
    signals = []
    overlays = {"hlines": [], "lines": []}

    last_close = float(raw["Close"].iloc[-1])

    # Trendline formations (triangles/wedges/broadening)
    tl = detect_trendlines(raw)
    if tl is not None and tl["pattern"] != "Channel/Other":
        upper, lower = boundary_now(raw, tl)
        d_upper = pct(last_close, upper)
        d_lower = pct(last_close, lower)

        if d_upper >= BREAK_PCT:
            signals.append(f"TRIGGER_{tl['pattern'].upper().replace(' ', '_')}_BREAKOUT")
        elif d_lower <= -BREAK_PCT:
            signals.append(f"TRIGGER_{tl['pattern'].upper().replace(' ', '_')}_BREAKDOWN")
        else:
            if abs(d_upper) <= NEAR_PCT or abs(d_lower) <= NEAR_PCT:
                signals.append(f"WATCH_{tl['pattern'].upper().replace(' ', '_')}_NEAR_BOUNDARY")

        w = raw.tail(WINDOW).copy().reset_index(drop=True)
        x = np.arange(len(w))
        overlays["lines"].append(("upper", tl["aU"] * x + tl["bU"]))
        overlays["lines"].append(("lower", tl["aL"] * x + tl["bL"]))

    # Rectangle
    rect = detect_rectangle(raw)
    if rect is not None:
        sup, res = rect["support"], rect["resistance"]
        if last_close >= res * (1 + BREAK_PCT/100):
            signals.append("TRIGGER_RECTANGLE_BREAKOUT")
        elif last_close <= sup * (1 - BREAK_PCT/100):
            signals.append("TRIGGER_RECTANGLE_BREAKDOWN")
        else:
            if abs(pct(last_close, res)) <= NEAR_PCT or abs(pct(last_close, sup)) <= NEAR_PCT:
                signals.append("WATCH_RECTANGLE_NEAR_EDGE")
        overlays["hlines"].extend([("support", sup), ("resistance", res)])

    # H&S
    hs = detect_hs_top(raw)
    if hs is not None:
        neck = hs["neck"]
        if last_close <= neck * (1 - BREAK_PCT/100):
            signals.append("TRIGGER_HS_TOP_BREAKDOWN")
        elif 0 <= pct(last_close, neck) <= NEAR_PCT:
            signals.append("WATCH_HS_TOP_NEAR_NECKLINE")
        overlays["hlines"].append(("hs_neckline", neck))

    inv = detect_inv_hs(raw)
    if inv is not None:
        neck = inv["neck"]
        if last_close >= neck * (1 + BREAK_PCT/100):
            signals.append("TRIGGER_INV_HS_BREAKOUT")
        elif 0 <= pct(neck, last_close) <= NEAR_PCT:
            signals.append("WATCH_INV_HS_NEAR_NECKLINE")
        overlays["hlines"].append(("inv_hs_neckline", neck))

    return sorted(set(signals)), overlays


# -------------------- State --------------------
def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# -------------------- Charts --------------------
def make_chart(raw: pd.DataFrame, title: str, outpath: str, overlays: dict):
    # Price line
    w = raw.tail(320).copy().reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(w["Date"], w["Close"])

    # Horizontal lines
    for _, y in overlays.get("hlines", []):
        plt.axhline(y)

    # Trendlines: draw on last WINDOW segment
    if overlays.get("lines"):
        w2 = raw.tail(WINDOW).copy().reset_index()
        dates2 = w2["Date"]
        for _, yarr in overlays["lines"]:
            plt.plot(dates2, yarr)

    plt.title(title[:180])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


# -------------------- Main --------------------
def main():
    if not should_run_now():
        print("Not scheduled time and FORCE_RUN!=1; exiting.")
        return

    ndx = get_nasdaq100_tickers()
    spx = get_sp500_tickers()
    custom = read_custom_tickers()

    ndx_set, spx_set, custom_set = set(ndx), set(spx), set(custom)
    tickers = sorted(set(ndx + spx + custom))

    print(f"Universe size: {len(tickers)} tickers (NDX={len(ndx)}, SPX={len(spx)}, CUSTOM={len(custom)})")

    data = download_chunked(tickers)

    today = now_local().date().isoformat()
    img_dir = os.path.join("docs", "img", today)
    os.makedirs(img_dir, exist_ok=True)

    prev_state = load_state()
    new_state = {}

    def universe(t):
        if t in custom_set:
            return "CUSTOM"
        if t in ndx_set:
            return "NDX"
        if t in spx_set:
            return "SPX"
        return "OTHER"

    # After-hours snapshot (custom only)
    movers = []
    if custom and CUSTOM_AFTERHOURS_ONLY:
        for t in custom:
            ah = after_hours_snapshot(t)
            if not math.isnan(ah["ah_change_pct"]):
                movers.append({"ticker": t, **ah})
    movers_df = pd.DataFrame(movers)

    rows = []

    # Scan tickers
    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue
        raw = data[t].dropna()
        if raw.empty or len(raw) < 200:
            continue

        sigs, overlays = compute_signals_for_ticker(raw)
        new_state[t] = sigs

        prev_sigs = set(prev_state.get(t, []))
        curr_sigs = set(sigs)

        new_signals = sorted(list(curr_sigs - prev_sigs))
        ended_signals = sorted(list(prev_sigs - curr_sigs))

        if curr_sigs or ended_signals:
            rows.append({
                "universe": universe(t),
                "ticker": t,
                "last_close": float(raw["Close"].iloc[-1]),
                "signals": sigs,
                "new": new_signals,
                "ended": ended_signals,
                "overlays": overlays
            })

    save_state(new_state)

    df = pd.DataFrame(rows)

    def is_trigger(sig): return sig.startswith("TRIGGER_")
    def is_watch(sig): return sig.startswith("WATCH_")

    if df.empty:
        triggers_new = df
        watchlist = df
        ended = df
    else:
        triggers_new = df[df["new"].apply(lambda xs: any(is_trigger(x) for x in xs))].copy()
        watchlist = df[df["signals"].apply(lambda xs: any(is_watch(x) for x in xs))].copy()
        ended = df[df["ended"].apply(lambda xs: len(xs) > 0)].copy()

    # Chart generation (caps per universe)
    charts_done = {"CUSTOM": 0, "NDX": 0, "SPX": 0, "OTHER": 0}
    chart_paths = {}

    if not df.empty:
        for uni in ["CUSTOM", "NDX", "SPX", "OTHER"]:
            sub = df[df["universe"] == uni].copy()
            if sub.empty:
                continue

            # prioritize NEW triggers, then triggers, then watches, then ended
            def prio(r):
                score = 0
                if any(is_trigger(x) for x in r["new"]): score += 4
                if any(is_trigger(x) for x in r["signals"]): score += 2
                if any(is_watch(x) for x in r["signals"]): score += 1
                if len(r["ended"]) > 0: score += 0.5
                return -score, r["ticker"]

            sub = sub.sort_values(by=["ticker"]).copy()
            sub["__k"] = sub.apply(prio, axis=1)
            sub = sub.sort_values(by="__k").drop(columns="__k")

            cap = MAX_CHARTS_PER_UNIVERSE if uni in ("NDX", "SPX") else 9999

            for _, r in sub.head(cap).iterrows():
                if not (r["signals"] or r["ended"]):
                    continue

                parts = []
                if r["new"]:
                    parts.append("NEW: " + ", ".join(r["new"]))
                if r["signals"]:
                    parts.append("NOW: " + ", ".join(r["signals"]))
                if r["ended"]:
                    parts.append("ENDED: " + ", ".join(r["ended"]))
                title = f"{r['ticker']} — " + " | ".join(parts)

                outpath = os.path.join(img_dir, f"{r['ticker']}.png")
                try:
                    raw2 = data[r["ticker"]].dropna()
                    make_chart(raw2, title, outpath, r["overlays"])
                    chart_paths[r["ticker"]] = outpath.replace("docs/", "")
                    charts_done[uni] += 1
                except Exception:
                    pass

    # -------------------- Build report markdown --------------------
    md = []
    md.append(f"# Daily Report — {today} (22:30 {TZ})\n")

    md.append("## After-hours snapshot (custom watchlist)\n")
    if movers_df.empty:
        md.append("_No extended-hours data available from source (or none computed)._ \n")
    else:
        movers_df = movers_df.sort_values("ah_change_pct", ascending=False)
        md.append(movers_df[["ticker","reg_close","last_px","ah_change_pct"]].to_markdown(index=False))
        md.append("\n")

    md.append("## NEW technical triggers (today)\n")
    if triggers_new.empty:
        md.append("_None detected._\n")
    else:
        for uni in ["CUSTOM", "NDX", "SPX"]:
            sub = triggers_new[triggers_new["universe"] == uni].copy()
            if sub.empty:
                continue
            md.append(f"### {uni}\n")
            for _, r in sub.sort_values(by=["ticker"]).iterrows():
                md.append(f"#### {r['ticker']} — {', '.join(r['new'])}\n")
                md.append(f"- Last close: {r['last_close']:.2f}\n")
                if r["ticker"] in chart_paths:
                    md.append(f"![{r['ticker']}]({chart_paths[r['ticker']]})\n")

    md.append("## Watchlist (near completion / near boundary)\n")
    if watchlist.empty:
        md.append("_None detected._\n")
    else:
        for uni in ["CUSTOM", "NDX", "SPX"]:
            sub = watchlist[watchlist["universe"] == uni].copy()
            if sub.empty:
                continue
            md.append(f"### {uni}\n")
            for _, r in sub.sort_values(by=["ticker"]).iterrows():
                watches = [x for x in r["signals"] if x.startswith("WATCH_")]
                md.append(f"#### {r['ticker']} — {', '.join(watches)}\n")
                md.append(f"- Last close: {r['last_close']:.2f}\n")
                if r["ticker"] in chart_paths:
                    md.append(f"![{r['ticker']}]({chart_paths[r['ticker']]})\n")

    md.append("## Ended / invalidated signals\n")
    if ended.empty:
        md.append("_None._\n")
    else:
        for uni in ["CUSTOM", "NDX", "SPX"]:
            sub = ended[ended["universe"] == uni].copy()
            if sub.empty:
                continue
            md.append(f"### {uni}\n")
            for _, r in sub.sort_values(by=["ticker"]).iterrows():
                md.append(f"- **{r['ticker']}** ended: {', '.join(r['ended'])}\n")

    os.makedirs("docs", exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md))
    with open(INDEX_MD, "w") as f:
        f.write("\n".join(md))

    print(f"Wrote {REPORT_MD} and updated {STATE_PATH}")


if __name__ == "__main__":
    main()
