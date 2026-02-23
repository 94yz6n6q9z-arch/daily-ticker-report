# tools/pattern_charts.py
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _slug(s: str) -> str:
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9\.\-_]+", "_", s)
    return s.strip("_")


def _last_valid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where OHLC missing (weekends/partial rows)
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    out = df.dropna(subset=cols).copy()
    return out


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    df = _last_valid_ohlc(df)
    if df.empty:
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = _true_range(high, low, close)
    if len(tr) < n:
        return pd.Series([np.nan] * len(tr), index=df.index)
    atr = tr.copy() * 0.0
    atr.iloc[: n - 1] = np.nan
    atr.iloc[n - 1] = tr.iloc[:n].mean()
    for i in range(n, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (n - 1) + tr.iloc[i]) / n
    return atr


def _pivots(arr: np.ndarray, w: int = 5, kind: str = "high") -> list[int]:
    piv = []
    for i in range(w, len(arr) - w):
        window = arr[i - w : i + w + 1]
        if kind == "high":
            if arr[i] == np.max(window) and np.sum(window == arr[i]) == 1:
                piv.append(i)
        else:
            if arr[i] == np.min(window) and np.sum(window == arr[i]) == 1:
                piv.append(i)
    return piv


def _annotate_hs_top(ax, close: np.ndarray, low: np.ndarray) -> Optional[float]:
    piv = _pivots(close, w=5, kind="high")[-12:]
    if len(piv) < 3:
        return None

    best = None
    for i in range(len(piv) - 2):
        a, b, c = piv[i], piv[i + 1], piv[i + 2]
        if close[b] > close[a] and close[b] > close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)

    if best is None:
        best = (piv[-3], piv[-2], piv[-1])

    ls, head, rs = best
    for idx, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([idx], [close[idx]], s=40)
        ax.annotate(label, (idx, close[idx]), xytext=(idx, close[idx] + 3),
                    arrowprops=dict(arrowstyle="->", lw=1))

    n1 = float(np.min(low[min(ls, head) : max(ls, head) + 1]))
    n2 = float(np.min(low[min(head, rs) : max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0

    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(len(close) - 1, neckline, " Neckline", va="bottom")
    return neckline


def _annotate_ihs(ax, close: np.ndarray, high: np.ndarray) -> Optional[float]:
    piv = _pivots(close, w=5, kind="low")[-12:]
    if len(piv) < 3:
        return None

    best = None
    for i in range(len(piv) - 2):
        a, b, c = piv[i], piv[i + 1], piv[i + 2]
        if close[b] < close[a] and close[b] < close[c]:
            if abs(close[a] - close[c]) / max(close[a], close[c]) < 0.12:
                best = (a, b, c)

    if best is None:
        best = (piv[-3], piv[-2], piv[-1])

    ls, head, rs = best
    for idx, label in [(ls, "LS"), (head, "H"), (rs, "RS")]:
        ax.scatter([idx], [close[idx]], s=40)
        ax.annotate(label, (idx, close[idx]), xytext=(idx, close[idx] - 4),
                    arrowprops=dict(arrowstyle="->", lw=1))

    n1 = float(np.max(high[min(ls, head) : max(ls, head) + 1]))
    n2 = float(np.max(high[min(head, rs) : max(head, rs) + 1]))
    neckline = (n1 + n2) / 2.0

    ax.axhline(neckline, linestyle="--", linewidth=1)
    ax.text(len(close) - 1, neckline, " Neckline", va="bottom")
    return neckline


def _placeholder(path: Path, title: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.75, title, fontsize=14, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.50, "Chart unavailable", fontsize=12, transform=ax.transAxes)
    ax.text(0.02, 0.30, f"Reason: {reason}", fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def ensure_annotated_chart(
    df: pd.DataFrame,
    ticker: str,
    signal: str,
    level: float,
    dist_atr: Optional[float],
    out_dir: Path = Path("docs/img"),
) -> str:
    """
    Always creates a chart file (real or placeholder) and returns relative markdown path: img/<file>.png
    Adds:
      - Trigger line
      - Confirm line (±0.5 ATR)
      - Pattern markings for HS_TOP / IHS (shoulders/head + neckline)
      - A small "trade prep" box (trigger/confirm/dist)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_slug(ticker)}_{_slug(signal)}.png"
    out_path = out_dir / fname

    try:
        df = _last_valid_ohlc(df)
        if df.empty or "Close" not in df.columns:
            raise RuntimeError("No valid OHLC data")

        close = df["Close"].astype(float).values
        high = df["High"].astype(float).values if "High" in df.columns else close
        low = df["Low"].astype(float).values if "Low" in df.columns else close

        atr = wilder_atr(df, 14)
        atr_last = float(atr.dropna().iloc[-1]) if len(atr.dropna()) else 0.0

        # Confirm line uses your rule: ±0.5 ATR past trigger
        direction = +1 if "BREAKOUT" in signal else -1 if "BREAKDOWN" in signal else 0
        confirm = level + direction * 0.5 * atr_last

        fig = plt.figure(figsize=(11, 5.5))
        ax = fig.add_subplot(111)
        ax.plot(close)
        ax.set_title(f"{ticker} — {signal}")

        # Trigger + confirm lines (no explicit colors; matplotlib defaults)
        ax.axhline(level, linestyle="-.", linewidth=1)
        ax.axhline(confirm, linestyle=":", linewidth=1)

        ax.text(len(close) - 1, level, " Trigger", va="bottom")
        ax.text(len(close) - 1, confirm, " Confirm (±0.5 ATR)", va="bottom")

        # Pattern markings
        if "HS_TOP" in signal:
            _annotate_hs_top(ax, close, low)
        if "IHS" in signal:
            _annotate_ihs(ax, close, high)

        # Latest close marker
        ax.scatter([len(close) - 1], [close[-1]], s=60)
        ax.annotate("Close", (len(close) - 1, close[-1]),
                    xytext=(len(close) - 12, close[-1] + 3),
                    arrowprops=dict(arrowstyle="->", lw=1))

        # Trade-prep box
        dist_txt = "" if dist_atr is None else f"{dist_atr:+.2f} ATR"
        box = (
            f"Trigger: {level:.2f}\n"
            f"Confirm: {confirm:.2f}\n"
            f"Dist: {dist_txt}"
        )
        ax.text(0.02, 0.02, box, transform=ax.transAxes,
                fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.6))

        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        # sanity: if file too small, treat as failed
        if not out_path.exists() or out_path.stat().st_size < 5000:
            raise RuntimeError("Chart file missing/too small after save")

    except Exception as e:
        _placeholder(out_path, f"{ticker} — {signal}", str(e))

    return f"img/{fname}"


def explain_early_callout_md(
    ticker: str,
    signal: str,
    close: float,
    level: float,
    atr14: float,
    dist_atr: float,
) -> str:
    """
    Deterministic explanation block for NEW early callouts.
    """
    direction = "breakout" if "BREAKOUT" in signal else "breakdown" if "BREAKDOWN" in signal else "move"
    confirm = level + (0.5 * atr14 if "BREAKOUT" in signal else -0.5 * atr14 if "BREAKDOWN" in signal else 0.0)

    parts = []
    parts.append(f"**{ticker} — what’s going on (trade prep)**")
    parts.append(f"- **Signal:** `{signal}` (early {direction}).")
    parts.append(f"- **Trigger level:** {level:.2f}. **Confirm level (±0.5 ATR):** {confirm:.2f}.")
    parts.append(f"- **Where we are now:** close {close:.2f} | distance {dist_atr:+.2f} ATR.")
    parts.append("- **How to use it:** treat the trigger as your “ready” line; only act on confirmation close or a retest/failure depending on your playbook.")
    return "\n".join(parts)
