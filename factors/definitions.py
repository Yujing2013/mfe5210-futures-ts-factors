from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log(a / b.replace(0, np.nan))


def momentum_120_20(g: pd.DataFrame) -> pd.Series:
    """12-1 style momentum (approx. 120 sessions lookback, skip 20)."""
    c = g["close"]
    return _safe_log_ratio(c.shift(20), c.shift(120))


def momentum_60_10(g: pd.DataFrame) -> pd.Series:
    c = g["close"]
    return _safe_log_ratio(c.shift(10), c.shift(60))


def vol_fade_60d(g: pd.DataFrame) -> pd.Series:
    """Negated 60-day realized vol (fade high-vol)."""
    r = g["close"].pct_change()
    rv = r.rolling(60, min_periods=20).std()
    return -rv


def gk_vol_level_5d(g: pd.DataFrame) -> pd.Series:
    """Garman-Klass range variance, 5-day mean (signal negated)."""
    h = g["high"].astype(float)
    lo = g["low"].astype(float)
    c = g["close"].astype(float)
    o = g["open"].astype(float)
    hl = np.log(h / lo.replace(0, np.nan)) ** 2
    co = np.log(c / o.replace(0, np.nan)) ** 2
    est = 0.5 * hl - (2 * np.log(2) - 1) * co
    return -est.rolling(5, min_periods=3).mean()


def volume_shock_neg(g: pd.DataFrame) -> pd.Series:
    """Negated volume z-score — fade volume shocks."""
    v = g["volume"].astype(float)
    vm = v.rolling(20, min_periods=10).mean()
    vs = v.rolling(20, min_periods=10).std()
    vz = (v - vm) / vs.replace(0, np.nan)
    return -vz


def oi_trend_5d(g: pd.DataFrame) -> pd.Series:
    h = g["hold"].astype(float)
    return h.pct_change(5)


def ret_skew_60d(g: pd.DataFrame) -> pd.Series:
    """60-day return skewness."""
    r = g["close"].pct_change()
    return r.rolling(60, min_periods=25).skew()


def trend_efficiency_20d(g: pd.DataFrame) -> pd.Series:
    """Efficiency ratio: |20d drift| / cumulative path length."""
    c = g["close"]
    r = c.pct_change()
    num = (c / c.shift(20) - 1).abs()
    den = r.abs().rolling(20, min_periods=10).sum()
    return num / den.replace(0, np.nan)


def overnight_minus_intraday(g: pd.DataFrame) -> pd.Series:
    """Overnight return minus intraday return."""
    c = g["close"]
    o = g["open"].astype(float)
    overnight = o / c.shift(1) - 1
    intraday = c / o.replace(0, np.nan) - 1
    return overnight - intraday


def range_breakout_20d(g: pd.DataFrame) -> pd.Series:
    """Close position within 20-day high-low range."""
    c = g["close"]
    hi = c.rolling(20, min_periods=10).max()
    lo = c.rolling(20, min_periods=10).min()
    return (c - lo) / (hi - lo).replace(0, np.nan)


def pv_corr_20d(g: pd.DataFrame) -> pd.Series:
    """20-day rolling corr between returns and log-volume changes."""
    r = g["close"].pct_change()
    dv = g["volume"].astype(float).pct_change()
    return r.rolling(20, min_periods=10).corr(dv)


def downside_upside_vol_ratio_20d(g: pd.DataFrame) -> pd.Series:
    """Ratio of downside vs upside return volatility."""
    r = g["close"].pct_change()
    neg = r.where(r < 0)
    pos = r.where(r > 0)
    sd_n = neg.rolling(20, min_periods=10).std()
    sd_p = pos.rolling(20, min_periods=10).std()
    return sd_n / sd_p.replace(0, np.nan)


def ret_autocorr_lag2_60d(g: pd.DataFrame) -> pd.Series:
    """Rolling correlation between returns and their 2-day lag."""
    r = g["close"].pct_change()
    return r.rolling(60, min_periods=30).corr(r.shift(2))


def amihud_illiq_20d(g: pd.DataFrame) -> pd.Series:
    """20-day average |return|/volume (Amihud-style illiquidity)."""
    r = g["close"].pct_change().abs()
    v = g["volume"].astype(float).replace(0, np.nan)
    return (r / v).rolling(20, min_periods=10).mean()


def momentum_252_21(g: pd.DataFrame) -> pd.Series:
    """Classic TSMOM horizon (≈252 sessions, skip 21)."""
    c = g["close"]
    return _safe_log_ratio(c.shift(21), c.shift(252))


def ret_kurt_60d(g: pd.DataFrame) -> pd.Series:
    """60-day excess kurtosis of returns."""
    r = g["close"].pct_change()
    return r.rolling(60, min_periods=30).kurt()


def cci_20d(g: pd.DataFrame) -> pd.Series:
    """Lite Commodity Channel Index on typical price."""
    h = g["high"].astype(float)
    lo = g["low"].astype(float)
    c = g["close"].astype(float)
    tp = (h + lo + c) / 3.0
    ma = tp.rolling(20, min_periods=10).mean()
    mad = (tp - ma).abs().rolling(20, min_periods=10).mean()
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def tsm_reversal_5d(g: pd.DataFrame) -> pd.Series:
    """Negated 5-day cumulative return — short horizon reversal tilt."""
    c = g["close"]
    past = c / c.shift(5) - 1.0
    return -past


def macd_spread_20_60(g: pd.DataFrame) -> pd.Series:
    """EMA(20)-EMA(60) MACD-style trend strength."""
    c = g["close"].astype(float)
    fast = c.ewm(span=20, adjust=False).mean()
    slow = c.ewm(span=60, adjust=False).mean()
    return fast - slow


def oi_accel_3d(g: pd.DataFrame) -> pd.Series:
    """First difference of 3-day pct change in open interest."""
    h = g["hold"].astype(float)
    r3 = h.pct_change(3)
    return r3.diff()


def vol_oscillator_5_20(g: pd.DataFrame) -> pd.Series:
    """Short vs long MA of volume minus one (distinct from shock z-score)."""
    v = g["volume"].astype(float)
    m5 = v.rolling(5, min_periods=3).mean()
    m20 = v.rolling(20, min_periods=10).mean()
    return m5 / m20.replace(0, np.nan) - 1.0


def overnight_gap_mean_5d(g: pd.DataFrame) -> pd.Series:
    """5-day mean overnight gap vs prior close."""
    c = g["close"]
    o = g["open"].astype(float)
    gap = o / c.shift(1).replace(0, np.nan) - 1.0
    return gap.rolling(5, min_periods=3).mean()


def up_day_fraction_20d(g: pd.DataFrame) -> pd.Series:
    """20-day fraction of up days minus 0.5."""
    r = g["close"].pct_change()
    up = (r > 0).astype(float)
    return up.rolling(20, min_periods=10).mean() - 0.5


def volume_price_cov_20d(g: pd.DataFrame) -> pd.Series:
    """20-day rolling covariance between returns and volume levels."""
    r = g["close"].pct_change()
    v = g["volume"].astype(float)
    return r.rolling(20, min_periods=10).cov(v)


def parkinson_hl_var_20d(g: pd.DataFrame) -> pd.Series:
    """20-day mean Parkinson high-low variance (complements GK)."""
    h = g["high"].astype(float)
    lo = g["low"].astype(float)
    lr = np.log(h / lo.replace(0, np.nan))
    return (lr**2).rolling(20, min_periods=10).mean()


def volume_rel_to_60max_20d(g: pd.DataFrame) -> pd.Series:
    """20-day mean volume relative to 60-day rolling max."""
    v = g["volume"].astype(float)
    mx = v.rolling(60, min_periods=25).max()
    ratio = v / mx.replace(0, np.nan)
    return ratio.rolling(20, min_periods=10).mean()


def return_sign_sum_10d(g: pd.DataFrame) -> pd.Series:
    """10-day mean return sign in [-1, 1]."""
    r = g["close"].pct_change()
    sgn = np.sign(r)
    return sgn.rolling(10, min_periods=5).mean()


def hl_range_pct_skew_20d(g: pd.DataFrame) -> pd.Series:
    """Skewness of intraday range (H-L)/C over 20 days."""
    c = g["close"].astype(float).replace(0, np.nan)
    hl = (g["high"].astype(float) - g["low"].astype(float)) / c
    return hl.rolling(20, min_periods=10).skew()


def carry_ma_dev_252(g: pd.DataFrame) -> pd.Series:
    """Close vs 252-day MA (level/carry proxy).

    Unlike `momentum_252_21` (log price vs older price), this measures distance to a
    slow moving average — often lower correlation with classic momentum in TS use.
    """
    c = g["close"].astype(float)
    ma = c.rolling(252, min_periods=100).mean()
    return c / ma.replace(0, np.nan) - 1.0


def vol_scaled_mom_60_10(g: pd.DataFrame) -> pd.Series:
    """Risk-adjusted 60–10 momentum: return / trailing return vol."""
    c = g["close"].astype(float)
    mom = c.shift(10) / c.shift(60) - 1.0
    rv = c.pct_change().rolling(60, min_periods=20).std()
    return mom / rv.replace(0, np.nan)


def oi_price_agree_20d(g: pd.DataFrame) -> pd.Series:
    """20-day mean sign(price change) × sign(OI change) in [-1, 1].

    Positive values mean price/OI movements align ('flow confirmation').
    Orthogonal to simple OI level trends.
    """
    rp = g["close"].astype(float).pct_change()
    ro = g["hold"].astype(float).pct_change()
    agree = np.sign(rp) * np.sign(ro)
    return agree.rolling(20, min_periods=10).mean()


def williams_pct_r_5d(g: pd.DataFrame) -> pd.Series:
    """Short Williams %R variant (centered): close location in 5d H/L band.

    Differs from `range_breakout_20d`: shorter window, uses true highs/lows.
    """
    c = g["close"].astype(float)
    h = g["high"].astype(float).rolling(5, min_periods=3).max()
    lo = g["low"].astype(float).rolling(5, min_periods=3).min()
    return (c - lo) / (h - lo).replace(0, np.nan) - 0.5


def body_streak_10d(g: pd.DataFrame) -> pd.Series:
    """10-day mean candle body sign(close-open) — streak bias."""

    body = np.sign(g["close"].astype(float) - g["open"].astype(float))
    return body.rolling(10, min_periods=5).mean()


def atr_norm_20d(g: pd.DataFrame) -> pd.Series:
    """ATR(20) divided by close — unit-free volatility level.

    Uses true range (captures gaps); complements GK / Parkinson proxies.
    """
    h = g["high"].astype(float)
    lo = g["low"].astype(float)
    c = g["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=10).mean()
    return atr / c.replace(0, np.nan)


def close_settle_spread_rel(g: pd.DataFrame) -> pd.Series:
    """(Close − settle) / settle when `settle` column exists (futures EOD wedge)."""
    if "settle" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    st = g["settle"].astype(float)
    c = g["close"].astype(float)
    return (c - st) / st.replace(0, np.nan)


def log_amount_per_vol_ma20(g: pd.DataFrame) -> pd.Series:
    """Log notional-per-lot smoothed 20d; upstream `amount` often in CNY 10k."""
    if "amount" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    am = g["amount"].astype(float) * 1e4
    v = g["volume"].astype(float).replace(0, np.nan)
    ratio = am / v
    ratio = ratio.where(ratio > 0, np.nan)
    return np.log(ratio).rolling(20, min_periods=10).mean()


def basis_dom_pct_change_10d(g: pd.DataFrame) -> pd.Series:
    """10-day diff of dominant-contract basis pct (needs `basis_dom_pct`)."""
    if "basis_dom_pct" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    return g["basis_dom_pct"].astype(float).diff(10)


def basis_dom_pct_ma20_dev(g: pd.DataFrame) -> pd.Series:
    """Basis minus its 20-day MA."""
    if "basis_dom_pct" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    b = g["basis_dom_pct"].astype(float)
    ma = b.rolling(20, min_periods=10).mean()
    return b - ma


def term_spread_pct_level(g: pd.DataFrame) -> pd.Series:
    """Term-structure spread level (near minus far scaled, needs column)."""
    if "term_spread_pct" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    return g["term_spread_pct"].astype(float)


def term_spread_change_20d(g: pd.DataFrame) -> pd.Series:
    """20-day change of `term_spread_pct`."""
    if "term_spread_pct" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    return g["term_spread_pct"].astype(float).diff(20)


def member_net_long_change_5d(g: pd.DataFrame) -> pd.Series:
    """5-day diff of `member_net_long_top20` leaderboard-style aggregate."""
    if "member_net_long_top20" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    return g["member_net_long_top20"].astype(float).diff(5)


def wsr_change_20d(g: pd.DataFrame) -> pd.Series:
    """20-day warehouse-receipt inventory change."""
    if "wsr_vol" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    return g["wsr_vol"].astype(float).diff(20)


def rel_mom_60d_vs_index(g: pd.DataFrame) -> pd.Series:
    """60-day log-return minus matching log-return of `index_close` (relative strength)."""
    if "index_close" not in g.columns:
        return pd.Series(np.nan, index=g.index)
    c = g["close"].astype(float)
    idx = g["index_close"].astype(float)
    return np.log(c / c.shift(60).replace(0, np.nan)) - np.log(
        idx / idx.shift(60).replace(0, np.nan)
    )
