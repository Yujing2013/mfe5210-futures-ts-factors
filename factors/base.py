from __future__ import annotations

import numpy as np
import pandas as pd


def per_symbol_signal(panel: pd.DataFrame, fn) -> pd.Series:
    """Apply `fn(symbol_df)->Series` per symbol on MultiIndex (date,symbol) panel."""
    parts: list[pd.Series] = []
    for sym, g in panel.groupby(level="symbol"):
        g = g.droplevel("symbol").sort_index()
        sig = fn(g)
        if not isinstance(sig, pd.Series):
            raise TypeError("fn must return Series indexed by date")
        sig = sig.reindex(g.index)
        idx = pd.MultiIndex.from_arrays([sig.index, np.full(len(sig), sym)], names=["date", "symbol"])
        parts.append(pd.Series(sig.values, index=idx, name="signal"))
    if not parts:
        return pd.Series(dtype=float, name="signal")
    out = pd.concat(parts).sort_index()
    return out


def ts_zscore(signal: pd.Series, window: int) -> pd.Series:
    """Rolling z-score independently within each symbol."""
    if signal.index.names != ["date", "symbol"]:
        raise ValueError("signal must have MultiIndex [date, symbol]")
    out_parts: list[pd.Series] = []
    for sym, s in signal.groupby(level="symbol"):
        s = s.droplevel("symbol").sort_index()
        mp = _zscore_min_periods(window)
        m = s.rolling(window, min_periods=mp).mean()
        v = s.rolling(window, min_periods=mp).std()
        z = (s - m) / v.replace(0, np.nan)
        idx = pd.MultiIndex.from_arrays([z.index, np.full(len(z), sym)], names=["date", "symbol"])
        out_parts.append(pd.Series(z.values, index=idx, name="signal"))
    return pd.concat(out_parts).sort_index()


def _zscore_min_periods(window: int) -> int:
    return max(10, window // 6)


def ts_winsor_roll_quantile(
    z: pd.Series,
    window: int,
    *,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> pd.Series:
    """
    Per-symbol rolling quantile clip: each observation is clipped to `[low_q, high_q]` of
    trailing `window` z-scores **for that symbol only** (pure time-series, no cross-section).
    """
    if z.index.names != ["date", "symbol"]:
        raise ValueError("signal must have MultiIndex [date, symbol]")
    min_p = max(15, window // 3)
    out_parts: list[pd.Series] = []
    for sym, s in z.groupby(level="symbol"):
        s = s.droplevel("symbol").sort_index()
        lo = s.rolling(window, min_periods=min_p).quantile(low_q)
        hi = s.rolling(window, min_periods=min_p).quantile(high_q)
        clipped = s.clip(lower=lo, upper=hi)
        idx = pd.MultiIndex.from_arrays(
            [clipped.index, np.full(len(clipped), sym)],
            names=["date", "symbol"],
        )
        out_parts.append(pd.Series(clipped.values, index=idx, name="signal"))
    return pd.concat(out_parts).sort_index()
