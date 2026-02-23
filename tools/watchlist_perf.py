# tools/watchlist_perf.py
from __future__ import annotations

from typing import Dict, List, Optional
import math
import pandas as pd
import yfinance as yf

GREEN = "#11823b"
RED = "#b91c1c"


def _fmt_num(x: Optional[float], decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:,.{decimals}f}"


def _fmt_pct(x: Optional[float], decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:+.{decimals}f}%"


def _fmt_ratio(x: Optional[float], decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{decimals}f}x"


def _color_pct(x: Optional[float], decimals: int = 1) -> str:
    s = _fmt_pct(x, decimals=decimals)
    if not s:
        return ""
    color = GREEN if x >= 0 else RED
    return f'<span style="color:{color};">{s}</span>'


def _calc_return_pct(close: pd.Series, n: int) -> Optional[float]:
    """Return percent (not fraction): +1.2 means +1.2%."""
    if close is None or len(close) < n + 1:
        return None
    a = float(close.iloc[-1])
    b = float(close.iloc[-(n + 1)])
    if b == 0 or math.isnan(a) or math.isnan(b):
        return None
    return (a / b - 1.0) * 100.0


def _calc_clv_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> Optional[float]:
    """CLV in percent: CLV[-1..+1] * 100."""
    if any(s is None or len(s) < 1 for s in (high, low, close)):
        return None
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])
    if math.isnan(h) or math.isnan(l) or math.isnan(c) or h <= l:
        return None
    clv = (2 * c - h - l) / (h - l)
    return clv * 100.0


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder ATR."""
    tr = _true_range(high, low, close)
    if len(tr) < n:
        return tr * float("nan")
    atr = tr.copy() * 0.0
    atr.iloc[: n - 1] = float("nan")
    atr.iloc[n - 1] = tr.iloc[:n].mean()
    for i in range(n, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (n - 1) + tr.iloc[i]) / n
    return atr


def _atr_delta_14d_pct(atr: pd.Series) -> Optional[float]:
    """% change vs 14 trading sessions ago."""
    if atr is None or len(atr) < 15:
        return None
    a_now = atr.iloc[-1]
    a_prev = atr.iloc[-15]
    if pd.isna(a_now) or pd.isna(a_prev) or a_prev == 0:
        return None
    return (float(a_now) / float(a_prev) - 1.0) * 100.0


def _vol_ratio(vol: pd.Series, window: int = 20) -> Optional[float]:
    if vol is None or len(vol) < window:
        return None
    v = float(vol.iloc[-1])
    avg = float(vol.iloc[-window:].mean())
    if avg == 0 or math.isnan(v) or math.isnan(avg):
        return None
    return v / avg


def _split_download(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                out[t] = df[t].copy()
    else:
        out[tickers[0]] = df.copy()
    return out


def _base_ticker(t: str) -> str:
    return t.split(".", 1)[0] if "." in t else t


def build_watchlist_performance_section_md(
    tickers_by_group: Dict[str, List[str]],
    atr_period: int = 14,
    ticker_labels: Optional[Dict[str, str]] = None,
    ticker_segment_rank: Optional[Dict[str, int]] = None,
) -> str:
    """
    Watchlist performance table.

    Columns (all 1 decimal):
      Ticker | Close | Day% | CLV% | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M

    Last 4 columns are color-coded green/red.

    Sorting:
      - first by segment rank (so Tankers/Refiners/etc cluster), then by 1D return desc
    """
    ticker_labels = ticker_labels or {}
    ticker_segment_rank = ticker_segment_rank or {}

    # Flatten tickers, preserve order, de-dupe
    all_tickers: List[str] = []
    for _, ts in tickers_by_group.items():
        all_tickers.extend(ts)
    seen = set()
    all_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]

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

    out: List[str] = []
    out.append("## 6) Watchlist performance (all tickers)")
    out.append("")
    out.append("Columns: **Close | Day% | CLV% | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M**")
    out.append("")

    for group, group_tickers in tickers_by_group.items():
        out.append(f"### {group}")
        out.append("")
        out.append("| Ticker | Close | Day% | CLV% | ATR(14) | ATR Δ14d | Vol/AvgVol(20) | 1D | 7D | 1M | 3M |")
        out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        rows = []
        for t in group_tickers:
            df = frames.get(t)
            disp = ticker_labels.get(t) or _base_ticker(t)

            if df is None or df.empty:
                rows.append((t, disp, None, None, None, None, None, None, None, None, None))
                continue

            cols = {c.lower(): c for c in df.columns}
            def col(name: str) -> pd.Series:
                key = cols.get(name)
                return df[key] if key else pd.Series(dtype=float)

            close = col("close")
            high = col("high")
            low = col("low")
            vol = col("volume")

            # Drop rows where OHLC is NaN (common on weekends/partial updates for some tickers)
            ohlc = pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": vol})
            ohlc = ohlc.dropna(subset=["High", "Low", "Close"])
            if ohlc.empty or len(ohlc) < 2:
                rows.append((t, disp, None, None, None, None, None, None, None, None, None))
                continue

            close_v = ohlc["Close"]
            high_v = ohlc["High"]
            low_v = ohlc["Low"]
            vol_v = ohlc["Volume"].fillna(0)

            last_close = float(close_v.iloc[-1])
            day_pct = _calc_return_pct(close_v, 1)
            clv_pct = _calc_clv_pct(high_v, low_v, close_v)

            atr_series = _wilder_atr(high_v, low_v, close_v, n=atr_period)
            atr_now = float(atr_series.iloc[-1]) if len(atr_series) else None
            if atr_now is not None and math.isnan(atr_now):
                atr_now = None
            atr_delta = _atr_delta_14d_pct(atr_series)

            vr = _vol_ratio(vol_v, 20)

            r1d = day_pct
            r7d = _calc_return_pct(close_v, 7)
            r1m = _calc_return_pct(close_v, 21)
            r3m = _calc_return_pct(close_v, 63)

            rows.append((t, disp, last_close, day_pct, clv_pct, atr_now, atr_delta, vr, r1d, r7d, r1m, r3m))

        def seg_rank(ticker_raw: str) -> int:
            return int(ticker_segment_rank.get(ticker_raw, 99))

        rows.sort(key=lambda x: (seg_rank(x[0]), -(x[8] if x[8] is not None else -9999.0)))

        for (t, disp, last_close, day_pct, clv_pct, atr_now, atr_delta, vr, r1d, r7d, r1m, r3m) in rows:
            out.append(
                "| {disp} | {close} | {day} | {clv} | {atr} | {atr_d} | {vr} | {c1} | {c7} | {c1m} | {c3m} |".format(
                    disp=disp,
                    close=_fmt_num(last_close, 1),
                    day=_fmt_pct(day_pct, 1),
                    clv=_fmt_pct(clv_pct, 1),
                    atr=_fmt_num(atr_now, 1),
                    atr_d=_fmt_pct(atr_delta, 1),
                    vr=_fmt_ratio(vr, 1),
                    c1=_color_pct(r1d, 1),
                    c7=_color_pct(r7d, 1),
                    c1m=_color_pct(r1m, 1),
                    c3m=_color_pct(r3m, 1),
                )
            )

        out.append("")

    return "\n".join(out)
