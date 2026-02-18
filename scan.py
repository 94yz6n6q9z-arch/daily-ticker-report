#!/usr/bin/env python3
import os, math
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

TZ = "Europe/Berlin"
RUN_AT_HOUR = 22
RUN_AT_MIN  = 30

LOOKBACK = "3y"
CHUNK = 25

PIVOT_W = 5
PATTERN_WINDOW = 260       # ~1 trading year
NEAR_PCT = 3.0             # near boundary
BREAK_PCT = 1.0            # trigger threshold (close beyond boundary)

def now_local():
    return datetime.now(pytz.timezone(TZ))

def should_run_now():
    t = now_local()
    return (t.hour == RUN_AT_HOUR) and (abs(t.minute - RUN_AT_MIN) <= 7)

def get_nasdaq100_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        if "Ticker" in t.columns:
            return [x.replace(".", "-") for x in t["Ticker"].astype(str).tolist()]
    return []

def get_sp500_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for t in tables:
        if "Symbol" in t.columns:
            return [x.replace(".", "-") for x in t["Symbol"].astype(str).tolist()]
    return []

def read_custom_tickers(path="config/tickers_custom.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        ticks = [x.strip() for x in f.readlines()]
    ticks = [x for x in ticks if x and not x.startswith("#")]
    return [x.replace(".", "-") for x in ticks]

def download_chunked(tickers):
    parts = []
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i+CHUNK]
        dfp = yf.download(
            part, period=LOOKBACK, interval="1d",
            group_by="ticker", auto_adjust=False,
            threads=True, progress=False
        )
        parts.append(dfp)
    return pd.concat(parts, axis=1)

def pct(a, b):
    return (a - b) / b * 100.0 if b else np.nan

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

def after_hours_snapshot(ticker: str):
    out = {"reg_close": np.nan, "last_px": np.nan, "ah_change_pct": np.nan}
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="2d", interval="1m", prepost=True)
        if h is None or h.empty:
            return out
        h = h.dropna()
        out["last_px"] = float(h["Close"].iloc[-1])
        idx = h.index

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

def detect_rectangle(df: pd.DataFrame, n=PATTERN_WINDOW):
    w = df.tail(n)
    if len(w) < 80: return None
    hi = w["High"].quantile(0.95)
    lo = w["Low"].quantile(0.05)
    rng = (hi - lo) / max(1e-9, lo)
    drift = pct(w["Close"].iloc[-1], w["Close"].iloc[0])
    if rng < 0.25 and abs(drift) < 20:
        return {"pattern":"Rectangle", "support":float(lo), "resistance":float(hi)}
    return None

def detect_trendlines(df: pd.DataFrame, n=PATTERN_WINDOW):
    w = df.tail(n).copy().reset_index(drop=True)
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
    aU,bU = upper
    aL,bL = lower
    last = len(w)-1
    early = max(0, last - 120)
    gap_now = line_y(aU,bU,last) - line_y(aL,bL,last)
    gap_then = line_y(aU,bU,early) - line_y(aL,bL,early)
    flatU = abs(aU) < 0.02
    flatL = abs(aL) < 0.02
    diverge  = gap_now > gap_then * 1.15
    converge = gap_now < gap_then * 0.85

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

    return {"pattern":patt, "aU":float(aU),"bU":float(bU),"aL":float(aL),"bL":float(bL),
            "gap_now":float(gap_now), "gap_then":float(gap_then)}

def boundary_now(df: pd.DataFrame, cls: dict):
    w = df.tail(PATTERN_WINDOW).copy().reset_index(drop=True)
    last = len(w)-1
    upper = line_y(cls["aU"], cls["bU"], last)
    lower = line_y(cls["aL"], cls["bL"], last)
    return float(upper), float(lower)

def detect_hs_top(df: pd.DataFrame, n=PATTERN_WINDOW):
    w = df.tail(n).copy().reset_index(drop=True)
    if len(w) < 120: return None
    close = w["Close"]
    low = w["Low"]
    dates = w["Date"]
    hi, _ = pivots(close, w=PIVOT_W)
    idx = hi.index.to_list()
    if len(idx) < 6: return None
    best=None
    for i in range(len(idx)-2):
        ls,hd,rs = idx[i], idx[i+1], idx[i+2]
        span = rs-ls
        if span < 25 or span > 220: continue
        ls_p,hd_p,rs_p = float(close.iloc[ls]), float(close.iloc[hd]), float(close.iloc[rs])
        if hd_p < max(ls_p, rs_p)*1.03: continue
        if abs(ls_p-rs_p)/((ls_p+rs_p)/2) > 0.15: continue
        t1 = int(low.iloc[ls:hd+1].idxmin())
        t2 = int(low.iloc[hd:rs+1].idxmin())
        t1p,t2p = float(low.loc[t1]), float(low.loc[t2])
        last = len(w)-1
        neck = t1p + (t2p - t1p) * ((last - t1) / max(1,(t2 - t1)))
        last_close = float(close.iloc[last])
        dist = pct(last_close, neck)
        best = {"pattern":"H&S Top", "ls":str(dates.iloc[ls].date()), "head":str(dates.iloc[hd].date()), "rs":str(dates.iloc[rs].date()),
                "neck":float(neck), "last":last_close, "dist_pct":float(dist)}
    return best

def detect_inv_hs(df: pd.DataFrame, n=PATTERN_WINDOW):
    w = df.tail(n).copy().reset_index(drop=True)
    if len(w) < 120: return None
    close = w["Close"]
    high = w["High"]
    dates = w["Date"]
    _, lo = pivots(close, w=PIVOT_W)
    idx = lo.index.to_list()
    if len(idx) < 6: return None
    best=None
    for i in range(len(idx)-2):
        ls,hd,rs = idx[i], idx[i+1], idx[i+2]
        span = rs-ls
        if span < 25 or span > 260: continue
        ls_p,hd_p,rs_p = float(close.iloc[ls]), float(close.iloc[hd]), float(close.iloc[rs])
        if hd_p > min(ls_p, rs_p)*0.97: continue
        if abs(ls_p-rs_p)/((ls_p+rs_p)/2) > 0.22: continue
        p1 = int(high.iloc[ls:hd+1].idxmax())
        p2 = int(high.iloc[hd:rs+1].idxmax())
        p1p,p2p = float(high.loc[p1]), float(high.loc[p2])
        last = len(w)-1
        neck = p1p + (p2p - p1p) * ((last - p1) / max(1,(p2 - p1)))
        last_close = float(close.iloc[last])
        dist = pct(neck, last_close)  # positive if below neckline
        best = {"pattern":"Inverse H&S", "ls":str(dates.iloc[ls].date()), "head":str(dates.iloc[hd].date()), "rs":str(dates.iloc[rs].date()),
                "neck":float(neck), "last":last_close, "dist_pct":float(dist)}
    return best

def make_chart(df: pd.DataFrame, title: str, outpath: str, overlays=None):
    plt.figure(figsize=(10,4))
    plt.plot(df["Date"], df["Close"])
    if overlays:
        for ov in overlays:
            if ov["type"] == "hline":
                plt.axhline(ov["y"])
            elif ov["type"] == "line":
                plt.plot(df["Date"], ov["y"])
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def main():
    if not should_run_now():
        print("Not 22:30 local; exiting.")
        return

    ndx = get_nasdaq100_tickers()
    spx = get_sp500_tickers()
    custom = read_custom_tickers()
    tickers = sorted(set(ndx + spx + custom))

    data = download_chunked(tickers)

    today = now_local().date().isoformat()
    img_dir = os.path.join("docs", "img", today)
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    movers = []

    custom_set = set(custom)
    ndx_set = set(ndx)
    spx_set = set(spx)

    def universe(t):
        if t in custom_set: return "CUSTOM"
        if t in ndx_set: return "NDX"
        if t in spx_set: return "SPX"
        return "OTHER"

    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue
        raw = data[t].dropna()
        if raw.empty:
            continue

        g = raw.reset_index().rename(columns={"Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        notes = []
        overlays = []

        cons = detect_trendlines(raw)
        if cons is not None and cons["pattern"] != "Channel/Other":
            upper, lower = boundary_now(raw, cons)
            last_close = float(g["Close"].iloc[-1])
            d_upper = pct(last_close, upper)
            d_lower = pct(last_close, lower)
            if d_upper >= BREAK_PCT:
                notes.append(f"{cons['pattern']} breakout ↑ ({d_upper:.1f}% above upper)")
            elif d_lower <= -BREAK_PCT:
                notes.append(f"{cons['pattern']} breakdown ↓ ({abs(d_lower):.1f}% below lower)")
            elif abs(d_upper) <= NEAR_PCT or abs(d_lower) <= NEAR_PCT:
                notes.append(f"{cons['pattern']} near boundary (upper {d_upper:.1f}%, lower {d_lower:.1f}%)")

            w = raw.tail(PATTERN_WINDOW).copy().reset_index()
            x = np.arange(len(w))
            overlays.append({"type":"line","y":cons["aU"]*x + cons["bU"]})
            overlays.append({"type":"line","y":cons["aL"]*x + cons["bL"]})

        rect = detect_rectangl_
