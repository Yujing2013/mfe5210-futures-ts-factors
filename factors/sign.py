"""Factor sign calibration (no lookahead).

Principles:
- Use **only training dates** (default first 50% of calendar dates): build daily portfolio
  returns from finalized signals times `fwd_close_ret`; annualized Sharpe maps to a global
  sign flip per factor (negative Sharpe -> multiply series by -1).
- After training, the sign is **fixed** so no future information revises calibration.
- OOS Sharpe (post-cutoff, same sizing rule) is stored for disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from mfe5210_cta.backtest.engine import portfolio_returns_from_signal, portfolio_sharpe


def train_cutoff_from_panel(panel: pd.DataFrame, train_frac: float) -> pd.Timestamp:
    """Last date belonging to earliest `train_frac` share of unique calendar dates."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    if len(dates) < 60:
        return pd.Timestamp(dates[-1])
    cutoff_idx = max(1, int(len(dates) * train_frac) - 1)
    return pd.Timestamp(dates[cutoff_idx])


@dataclass
class SignCalibration:
    signs: Dict[str, int]
    train_end: pd.Timestamp
    train_sharpes: Dict[str, float]
    oos_sharpes: Dict[str, float]


def calibrate_signs(
    signals: Dict[str, pd.Series],
    panel: pd.DataFrame,
    *,
    train_frac: float = 0.5,
    cross_section_demean: bool = True,
) -> SignCalibration:
    """Return ±1 sign map from training-window portfolio Sharpe (per factor).

    Computes OOS Sharpe after cutoff for diagnostics only."""
    if not signals:
        return SignCalibration({}, pd.Timestamp("NaT"), {}, {})
    cutoff = train_cutoff_from_panel(panel, train_frac)

    signs: Dict[str, int] = {}
    train_sh: Dict[str, float] = {}
    oos_sh: Dict[str, float] = {}
    for name, sig in signals.items():
        pnl = portfolio_returns_from_signal(
            sig, panel, cross_section_demean=cross_section_demean
        )
        train = pnl.loc[:cutoff]
        oos = pnl.loc[cutoff + pd.Timedelta(days=1) :]
        s_train_raw = portfolio_sharpe(train)
        s_oos_raw = portfolio_sharpe(oos)
        sign = -1 if (s_train_raw == s_train_raw and s_train_raw < 0.0) else 1
        signs[name] = sign
        s_train_signed = (
            sign * s_train_raw if s_train_raw == s_train_raw else float("nan")
        )
        s_oos_signed = (
            sign * s_oos_raw if s_oos_raw == s_oos_raw else float("nan")
        )
        train_sh[name] = float(s_train_signed)
        oos_sh[name] = float(s_oos_signed)

    return SignCalibration(signs=signs, train_end=cutoff, train_sharpes=train_sh, oos_sharpes=oos_sh)


def apply_signs(
    signals: Dict[str, pd.Series], signs: Dict[str, int]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for k, s in signals.items():
        sign = signs.get(k, 1)
        out[k] = s if sign == 1 else (s * -1.0)
    return out
