from __future__ import annotations

import numpy as np
import pandas as pd


def per_symbol_daily_pnl(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    signal_lag: int = 0,
    cross_section_demean: bool = True,
) -> pd.Series:
    """MultiIndex (date, symbol)，单品种日度 leg：sig × fwd_close_ret（可关截面去均值）。"""
    if "fwd_close_ret" not in panel.columns:
        raise ValueError("panel must contain fwd_close_ret")
    aligned = panel[["fwd_close_ret"]].copy()
    sig = signal.copy()
    if signal_lag:
        sig = sig.groupby(level="symbol").shift(signal_lag)
    aligned["sig"] = sig
    if cross_section_demean:
        aligned["sig"] = aligned["sig"] - aligned["sig"].groupby(level="date").transform("mean")
    aligned["pnl"] = aligned["sig"] * aligned["fwd_close_ret"]
    return aligned["pnl"]


def portfolio_returns_from_signal(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    signal_lag: int = 0,
    cross_section_demean: bool = True,
) -> pd.Series:
    """
    时间序列组合：signal × fwd_close_ret 在截面上等权平均。
    cross_section_demean=True 时，每日对信号做截面去均值（仅移除共同多头/空头倾斜），
    弱化商品指数共同因子导致的因子间 PnL 伪完全相关，便于满足作业对相关性上限的要求。
    """
    return (
        per_symbol_daily_pnl(
            signal,
            panel,
            signal_lag=signal_lag,
            cross_section_demean=cross_section_demean,
        )
        .groupby(level="date")
        .mean()
        .sort_index()
    )


def per_symbol_sharpes_from_signal(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    cross_section_demean: bool = False,
    periods: int = 252,
) -> dict[str, float]:
    """
    单因子、按品种：对 leg 日收益分别算年化 Sharpe。
    cross_section_demean=False（默认）时为「纯时序单品种」：当日仅用该品种 finalize 后信号 × 自身 fwd 收益。
    """
    pnl = per_symbol_daily_pnl(
        signal, panel, cross_section_demean=cross_section_demean
    )
    wide = pnl.unstack("symbol")
    return {
        str(sym): portfolio_sharpe(wide[sym], periods=periods) for sym in wide.columns
    }


def portfolio_sharpe(daily_portfolio_ret: pd.Series, periods: int = 252) -> float:
    """年化 Sharpe：sqrt(periods)*mean/std，输入为单因子组合日收益序列。"""
    x = daily_portfolio_ret.dropna()
    if len(x) < max(30, periods // 4):
        return float("nan")
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(periods) * mu / sd)


def daily_turnover_from_signal(
    signal: pd.Series,
    *,
    signal_lag: int = 0,
    cross_section_demean: bool = True,
) -> pd.Series:
    """
    截面权重按 |signal| 归一化后，组合换手 = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|（与日度 PnL 构造一致）。
    """
    sig = signal.copy()
    if signal_lag:
        sig = sig.groupby(level="symbol").shift(signal_lag)
    wide = sig.unstack("symbol")
    if cross_section_demean:
        wide = wide.sub(wide.mean(axis=1), axis=0)
    denom = wide.abs().sum(axis=1)
    w = wide.div(denom.replace(0, np.nan), axis=0).fillna(0.0)
    t = 0.5 * w.diff().abs().sum(axis=1)
    return t.sort_index()


def max_drawdown(daily_returns: pd.Series) -> float:
    """基于日复利的最大回撤（最小累计净值相对历史峰值的相对跌幅，为负数或 0）。"""
    r = daily_returns.astype(float).fillna(0.0)
    if r.empty:
        return float("nan")
    cum = (1.0 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def equity_and_underwater_series(daily_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    由单因子组合日收益得到累计净值（复利）与水下回撤序列（cum / peak - 1）。
    与 max_drawdown 使用同一复利定义；丢弃日收益中的 NaN 行后再累计。
    """
    r = daily_returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    equity = (1.0 + r).cumprod()
    underwater = equity / equity.cummax() - 1.0
    return equity, underwater


def net_returns_after_linear_cost(
    gross_daily_ret: pd.Series,
    daily_turnover: pd.Series,
    *,
    cost_bps_per_unit_turnover: float,
) -> pd.Series:
    """net = gross - (cost_bps/1e4) * turnover；单位换手上的成本以 bps 计。"""
    rate = cost_bps_per_unit_turnover / 10000.0
    g, t = gross_daily_ret.align(daily_turnover, join="inner")
    return g - rate * t.fillna(0.0)


def factor_research_metrics(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    cross_section_demean: bool = True,
    cost_bps: float = 0.0,
    periods: int = 252,
) -> dict:
    """单因子的 gross Sharpe、换手、回撤及线性成本下 net Sharpe。"""
    gross = portfolio_returns_from_signal(
        signal, panel, cross_section_demean=cross_section_demean
    )
    turn = daily_turnover_from_signal(
        signal, cross_section_demean=cross_section_demean
    )
    net = net_returns_after_linear_cost(
        gross,
        turn,
        cost_bps_per_unit_turnover=cost_bps,
    )
    return {
        "sharpe_gross": portfolio_sharpe(gross, periods=periods),
        "sharpe_net": portfolio_sharpe(net, periods=periods),
        "max_drawdown_gross": max_drawdown(gross),
        "max_drawdown_net": max_drawdown(net),
        "mean_daily_turnover": float(turn.mean()) if len(turn) else float("nan"),
    }
