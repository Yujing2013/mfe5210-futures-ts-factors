from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pandas as pd

from mfe5210_cta.factors import definitions as d
from mfe5210_cta.factors.base import per_symbol_signal, ts_winsor_roll_quantile, ts_zscore

SignalFn = Callable[[pd.DataFrame], pd.Series]

# Baseline universe (generic OHLCV+OI); `raw_factor_entries` appends column-gated extras
BASE_RAW_FACTOR_REGISTRY: List[Tuple[str, SignalFn]] = [
    ("momentum_120_20", d.momentum_120_20),
    ("momentum_60_10", d.momentum_60_10),
    ("vol_fade_60d", d.vol_fade_60d),
    ("gk_vol_level_5d", d.gk_vol_level_5d),
    ("volume_shock_neg", d.volume_shock_neg),
    ("oi_trend_5d", d.oi_trend_5d),
    ("ret_skew_60d", d.ret_skew_60d),
    ("trend_efficiency_20d", d.trend_efficiency_20d),
    ("overnight_intraday_spread", d.overnight_minus_intraday),
    ("range_breakout_20d", d.range_breakout_20d),
    ("pv_corr_20d", d.pv_corr_20d),
    ("down_up_vol_ratio_20d", d.downside_upside_vol_ratio_20d),
    ("ret_autocorr_lag2_60d", d.ret_autocorr_lag2_60d),
    ("amihud_illiq_20d", d.amihud_illiq_20d),
    ("momentum_252_21", d.momentum_252_21),
    ("ret_kurt_60d", d.ret_kurt_60d),
    ("oi_accel_3d", d.oi_accel_3d),
    ("overnight_gap_mean_5d", d.overnight_gap_mean_5d),
    ("volume_rel_to_60max_20d", d.volume_rel_to_60max_20d),
    ("hl_range_pct_skew_20d", d.hl_range_pct_skew_20d),
    ("carry_ma_dev_252", d.carry_ma_dev_252),
    ("tsm_reversal_5d", d.tsm_reversal_5d),
    ("vol_scaled_mom_60_10", d.vol_scaled_mom_60_10),
    ("oi_price_agree_20d", d.oi_price_agree_20d),
    ("williams_pct_r_5d", d.williams_pct_r_5d),
    ("body_streak_10d", d.body_streak_10d),
    ("atr_norm_20d", d.atr_norm_20d),
]

# Alias kept for notebooks / downstream imports
RAW_FACTOR_REGISTRY = BASE_RAW_FACTOR_REGISTRY


def _nn(col: str, panel: pd.DataFrame) -> int:
    if col not in panel.columns:
        return 0
    return int(panel[col].notna().sum())


def raw_factor_entries(panel: pd.DataFrame) -> List[Tuple[str, SignalFn]]:
    rows = list(BASE_RAW_FACTOR_REGISTRY)
    if "settle" in panel.columns and panel["settle"].notna().any():
        if float(panel["settle"].fillna(0).abs().sum()) > 0:
            rows.append(("close_settle_spread_rel", d.close_settle_spread_rel))
    if "amount" in panel.columns and panel["amount"].notna().any():
        rows.append(("log_amount_per_vol_ma20", d.log_amount_per_vol_ma20))
    if "basis_dom_pct" in panel.columns and _nn("basis_dom_pct", panel) >= 50:
        rows.append(("basis_dom_pct_change_10d", d.basis_dom_pct_change_10d))
        rows.append(("basis_dom_pct_ma20_dev", d.basis_dom_pct_ma20_dev))
    if "term_spread_pct" in panel.columns and _nn("term_spread_pct", panel) >= 50:
        rows.append(("term_spread_pct_level", d.term_spread_pct_level))
        rows.append(("term_spread_change_20d", d.term_spread_change_20d))
    if "member_net_long_top20" in panel.columns and _nn("member_net_long_top20", panel) >= 100:
        rows.append(("member_net_long_change_5d", d.member_net_long_change_5d))
    if "wsr_vol" in panel.columns and _nn("wsr_vol", panel) >= 100:
        rows.append(("wsr_change_20d", d.wsr_change_20d))
    if "index_close" in panel.columns and _nn("index_close", panel) >= 100:
        rows.append(("rel_mom_60d_vs_index", d.rel_mom_60d_vs_index))
    return rows


def build_all_raw_signals(panel: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for name, fn in raw_factor_entries(panel):
        out[name] = per_symbol_signal(panel, fn)
    return out


def finalize_signals(
    raw: Dict[str, pd.Series],
    z_window: int,
    *,
    winsor_window: int | None = None,
    winsor_low_q: float = 0.05,
    winsor_high_q: float = 0.95,
) -> Dict[str, pd.Series]:
    """Rolling z-score per symbol followed by intra-symbol rolling quantile clip."""
    ww = winsor_window if winsor_window is not None else z_window
    finalized: Dict[str, pd.Series] = {}
    for k, s in raw.items():
        z = ts_zscore(s, z_window)
        finalized[k] = ts_winsor_roll_quantile(
            z, ww, low_q=winsor_low_q, high_q=winsor_high_q
        )
    return finalized
