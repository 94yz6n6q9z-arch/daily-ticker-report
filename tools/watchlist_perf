# watchlist_perf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import pandas as pd
import yfinance as yf


@dataclass
class WLRow:
    ticker: str
    close: Optional[float]
    clv_pct: Optional[float]          # CLV * 100
    atr14: Optional[float]            # ATR in price units
    atr14_chg_pct: Optional[float]    # % change vs 14 trading days ago
    vol_ratio: Optional[float]        # Volume / AvgVol(20)
    r_1d: Optional[float]
    r_7d: Optional[float]
    r_1m: Optional[float]
    r_3m: Optional[float]


GREEN = "#11823b"
RED = "#b91c1c"


def _is_nan(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x)))


def fmt_num_1(x: Optional[float]) -> str:
    if _is_nan(x):
        return ""
    return f"{x:,.1f}"


def fmt_pct_1(x: Optional[float]) -> str:
    """x is fractional return (e.g., 0.0123 -> +1.2%)"""
    if _is_nan(x):
        return ""
    return f"{x*100:+.1f}%"


def fmt_colored_pct_1(x: Optional[float]) -> str:
    if _is_nan(x):
        return ""
    color = GREEN if x >= 0 else RED
    return f'<span style="color:{color};">{x*100:+.1f}%</span>'


def fmt_ratio_1(x: Optional[float]) -> str:
    if _is_nan(x):
        return ""
    return f"{x:.1f}x"


def _calc_return(close: pd.Series, n: int) -> Optional[float]:
    if close is None or len(close) < n + 1:
        return None
    latest = close.iloc[-1]
    prev = close.iloc[-(n + 1)]
    if pd.isna(latest) or pd.isna(prev) or prev == 0:
        return None
    return float(latest / prev - 1.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR (RMA). Returns full ATR series aligned to df index."""
    tr = _true_range(df)
    atr = tr.rolling(n).mean()  # seed with SMA
    # Wilder smoothing
    for i in range(n, len(tr)):
        if pd.isna(atr.iloc[i - 1]):
            continue
        atr.iloc[i] = (atr.iloc[i - 1] * (n - 1) + tr.iloc[i]) / n
    return atr


def _calc_vol_ratio(vol: pd.Series, window: int = 20) -> Optional[float]:
    if vol is None or len(vol) < window:
        return None
    v = vol.iloc[-1]
    avg = vol.iloc[-window:].mean()
    if pd.isna(v) or pd.isna(avg) or avg == 0:
        return None
    return float(v / avg)


def _calc_clv_pct(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    if any(pd.isna(x) for x in (h, l, c)) or h <= l:
        return None
    clv = (2 * c - h - l) / (h - l)
    return float(clv * 100.0)


def _split_download(data: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                sub = data[t].copy()
                # Standardize columns to OHLCV capitalized
                sub = sub.rename(
                    columns={c: c.capitalize() for c in sub.columns}
                )
                out[t] = sub
    else:
        # single ticker
        sub = data.copy()
        sub = sub.rename(columns={c: c.capitalize() for c in sub.columns})
        out[tickers[0]] = sub
    return out


def build_watchlist_performance_section_md(
    tickers_by_group: Dict[str, List[str]],
) -> str:
    """
    Build grouped watchlist performance tables with columns:
    Ticker | Close | CLV (%) | ATR(14) | ATR Δ14d (%) | Vol/AvgVol(20) | 1D | 7D | 1M | 3M
    Returns (last 4 cols) are color-coded.
    All figures are 1 decimal (or 1 decimal %).
    """
    # Flatten and de-dupe
    all_tickers: List[str] = []
    for ts in tickers_by_group.values():
        all_tickers.extend(ts)
    seen = set()
    all_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]

    # Need enough history for ATR and 3M returns.
    raw = yf.download(
        tickers=all_tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    frames = _split_download(raw, all_tickers)

    rows_by_group: Dict[str, List[WLRow]] = {g: [] for g in tickers_by_group.keys()}

    for group, group_tickers in tickers_by_group.items():
        for t in group_tickers:
            df = frames.get(t)
            if df is None or df.empty:
                rows_by_group[group].append(WLRow(t, None, None, None, None, None, None, None, None, None))
                continue

            # Ensure required cols exist
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in df.columns:
                    # Missing data from yfinance
                    rows_by_group[group].append(WLRow(t, None, None, None, None, None, None, None, None, None))
                    df = None
                    break
            if df is None:
                continue

            close = df["Close"]
            vol = df["Volume"]

            r1d = _calc_return(close, 1)
            r7d = _calc_return(close, 7)
            r1m = _calc_return(close, 21)
            r3m = _calc_return(close, 63)

            clv_pct = _calc_clv_pct(df)

            atr_series = _wilder_atr(df, 14)
            atr_now = float(atr_series.iloc[-1]) if len(atr_series) else None
            # ATR value 14 trading days ago (index -15)
            atr_prev = None
            if len(atr_series) >= 15 and not pd.isna(atr_series.iloc[-15]):
                atr_prev = float(atr_series.iloc[-15])
            atr_chg_pct = None
            if atr_now is not None and atr_prev is not None and atr_prev != 0:
                atr_chg_pct = float(atr_now / atr_prev - 1.0)

            vol_ratio = _calc_vol_ratio(vol, 20)

            last_close = float(close.iloc[-1]) if len(close) else None

            rows_by_group[group].append(
                WLRow(
                    ticker=t,
                    close=last_close,
                    clv_pct=clv_pct,
                    atr14=atr_now,
                    atr14_chg_pct=atr_chg_pct,
                    vol_ratio=vol_ratio,
                    r_1d=r1d,
                    r_7d=r7d,
                    r_1m=r1m,
                    r_3m=r3m,
                )
            )

        # Sort by 1D return desc
        rows_by_group[group].sort(key=lambda r: (r.r_1d if r.r_1d is not None else -999), reverse=True)

    out: List[str] = []
    out.append("## 6) Watchlist performance (all tickers)")
    out.append("")
    out.append("Columns: **Close | CLV (%) | ATR(14) | ATR Δ14d (%) | Vol/AvgVol(20) | 1D | 7D | 1M | 3M**")
    out.append("")

    for group, rows in rows_by_group.items():
        out.append(f"### {group}")
        out.append("")
        out.append("| Ticker | Close | CLV | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M |")
        out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            out.append(
                "| {t} | {close} | {clv} | {atr} | {atrchg} | {vr} | {d1} | {d7} | {m1} | {m3} |".format(
                    t=r.ticker,
                    close=fmt_num_1(r.close),
                    clv=("" if _is_nan(r.clv_pct) else f"{r.clv_pct:+.1f}%"),
                    atr=fmt_num_1(r.atr14),
                    atrchg=fmt_pct_1(r.atr14_chg_pct),
                    vr=fmt_ratio_1(r.vol_ratio),
                    d1=fmt_colored_pct_1(r.r_1d),
                    d7=fmt_colored_pct_1(r.r_7d),
                    m1=fmt_colored_pct_1(r.r_1m),
                    m3=fmt_colored_pct_1(r.r_3m),
                )
            )
        out.append("")

    return "\n".join(out) + "\n"
