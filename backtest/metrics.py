from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mfe5210_cta.backtest.engine import portfolio_sharpe, portfolio_returns_from_signal


def market_neutralize_returns(ret_mtx: pd.DataFrame, market_ret: pd.Series) -> pd.DataFrame:
    """将各列因子收益对市场（如全品种等权 forward 收益）做一元回归取残差，用于相关性报告。

    数学：对每列 y 估计 OLS 截距+斜率 y = α + β·m + ε，返回 ε（保留原始时间索引）。
    β 用全样本协方差/方差；不重估 α 截距是因为常数项不影响 corr。
    """
    m = market_ret.reindex(ret_mtx.index).astype(float)
    out = {}
    for c in ret_mtx.columns:
        y = ret_mtx[c].astype(float)
        df = pd.concat([y, m], axis=1).dropna()
        if len(df) < 50 or df.iloc[:, 1].var() == 0:
            out[c] = y
            continue
        beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
        out[c] = y - beta * m
    return pd.DataFrame(out)


def market_beta_per_factor(
    ret_mtx: pd.DataFrame, market_ret: pd.Series
) -> Dict[str, float]:
    """各因子相对市场的 β：cov(PnL, mkt)/var(mkt)。仅作披露用。"""
    m = market_ret.reindex(ret_mtx.index).astype(float)
    out: Dict[str, float] = {}
    for c in ret_mtx.columns:
        df = pd.concat([ret_mtx[c].astype(float), m], axis=1).dropna()
        if len(df) < 50 or df.iloc[:, 1].var() == 0:
            out[c] = float("nan")
            continue
        out[c] = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
    return out


def pnl_residual_correlation_matrix(
    factor_signals: Dict[str, pd.Series],
    panel: pd.DataFrame,
    *,
    cross_section_demean: bool = False,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, float]]:
    """因子组合 PnL 在市场中性化后做相关。

    步骤：
      1. 对每个因子用 portfolio_returns_from_signal 得到日 PnL（默认 *不做*
         CS-demean，保留 TS 因子的自然形态）。
      2. 市场收益取「全品种等权 fwd_close_ret 跨日均」。
      3. 各因子 PnL 对市场做 OLS 一元回归，取残差 ε。
      4. 在 ε 矩阵上做 Pearson 相关。

    ``end_date`` 若给定，则仅用 ``index <= end_date`` 的交易日（样本内选因子等）。
    ``start_date`` 若给定，则仅用 ``index >= start_date``（纯 OOS 段相关矩阵）。
    """
    cols: Dict[str, pd.Series] = {}
    for name, sig in factor_signals.items():
        cols[name] = portfolio_returns_from_signal(
            sig, panel, cross_section_demean=cross_section_demean
        )
    ret_mtx = pd.DataFrame(cols).sort_index()
    if start_date is not None:
        st = pd.Timestamp(start_date)
        ret_mtx = ret_mtx.loc[ret_mtx.index >= st]
    if end_date is not None:
        et = pd.Timestamp(end_date)
        ret_mtx = ret_mtx.loc[ret_mtx.index <= et]
    mkt = panel["fwd_close_ret"].groupby(level="date").mean().reindex(ret_mtx.index)
    resid = market_neutralize_returns(ret_mtx, mkt)
    betas = market_beta_per_factor(ret_mtx, mkt)
    corr = resid.dropna(how="all").corr(method="pearson")
    return corr, ret_mtx, mkt, betas


def signal_correlation_matrix(factor_signals: Dict[str, pd.Series], min_obs: int = 200) -> pd.DataFrame:
    """
    因子间相似度：对每个品种分别计算两因子（已finalize）信号的时间序列 Pearson 相关，
    再在品种间取平均。避免商品指数共同因子导致的组合 PnL 伪高相关。
    """
    names = list(factor_signals.keys())
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    syms = factor_signals[names[0]].index.get_level_values("symbol").unique()
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            corrs: list[float] = []
            for sym in syms:
                sa = factor_signals[a].xs(sym, level="symbol").dropna()
                sb = factor_signals[b].xs(sym, level="symbol").dropna()
                al, bl = sa.align(sb, join="inner")
                if len(al) >= min_obs:
                    corrs.append(float(al.corr(bl)))
            v = float(np.nanmean(corrs)) if corrs else float("nan")
            mat.loc[a, b] = v
            mat.loc[b, a] = v
    return mat


def greedy_from_corr_matrix(
    corr: pd.DataFrame,
    sharpes: Dict[str, float],
    max_corr: float = 0.5,
) -> List[str]:
    names = sorted(corr.columns, key=lambda n: sharpes.get(n, float("-inf")), reverse=True)
    picked: List[str] = []
    for cand in names:
        if cand not in corr.columns:
            continue
        ok = True
        for p in picked:
            r = corr.loc[cand, p]
            if np.isnan(r) or abs(float(r)) > max_corr:
                ok = False
                break
        if ok:
            picked.append(cand)
    return picked


def build_factor_return_matrix(
    factor_signals: Dict[str, pd.Series], panel: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    cols = {}
    for name, sig in factor_signals.items():
        cols[name] = portfolio_returns_from_signal(sig, panel, **kwargs)
    df = pd.DataFrame(cols).sort_index()
    return df


def market_forward_return(panel: pd.DataFrame) -> pd.Series:
    return panel["fwd_close_ret"].groupby(level="date").mean()


def corr_matrix(ret_mtx: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return ret_mtx.dropna(how="all").corr(method=method)


def max_offdiag_corr(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return float("nan")
    c = corr.copy().values.astype(float)
    np.fill_diagonal(c, np.nan)
    if np.all(np.isnan(c)):
        return float("nan")
    return float(np.nanmax(np.abs(c)))


def greedy_factor_subset(
    ret_mtx: pd.DataFrame,
    sharpes: Dict[str, float],
    max_corr: float = 0.5,
    ordered_names: Iterable[str] | None = None,
) -> List[str]:
    """
    按 Sharpe 从高到低贪心入选：与已选因子的相关（绝对值）均需 <= max_corr。
    """
    names = list(ordered_names) if ordered_names is not None else list(ret_mtx.columns)
    names = sorted(names, key=lambda n: sharpes.get(n, float("-inf")), reverse=True)
    picked: List[str] = []
    for cand in names:
        if cand not in ret_mtx.columns:
            continue
        ok = True
        for p in picked:
            s = ret_mtx[[cand, p]].dropna()
            if len(s) < 20:
                ok = False
                break
            r = s[cand].corr(s[p])
            if np.isnan(r) or abs(r) > max_corr:
                ok = False
                break
        if ok:
            picked.append(cand)
    return picked


def save_corr_heatmap(corr: pd.DataFrame, path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
