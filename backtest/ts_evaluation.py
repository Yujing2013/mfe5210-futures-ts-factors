"""TS 因子标准评价模块。

涵盖：
- per-symbol IC（Pearson/Spearman）
- IC IR（跨品种均值/标准差）
- TS Sharpe（**不做** CS-demean，单品种独立 sig×ret 后跨品种平均）
- t-statistic（pooled by symbol，使用每品种 mean/SE 后跨品种平均；
  亦提供 pooled OLS 版本）
- IC 衰减（lag = 1,2,3,5,10,20）
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _per_symbol_panel(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_symbol_obs: int = 60,
) -> Iterable[tuple[str, pd.DataFrame]]:
    """对齐 signal 与 panel['fwd_close_ret'] 后按品种产出 (sym, df[sig, ret])。"""
    if signal.index.names != ["date", "symbol"]:
        raise ValueError("signal must have MultiIndex [date, symbol]")
    if "fwd_close_ret" not in panel.columns:
        raise ValueError("panel must contain fwd_close_ret")
    aligned = pd.DataFrame({"sig": signal, "ret": panel["fwd_close_ret"]})
    for sym, g in aligned.groupby(level="symbol"):
        gg = g.droplevel("symbol").dropna()
        if len(gg) >= min_symbol_obs:
            yield str(sym), gg


def per_symbol_ic(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    method: str = "pearson",
    lag: int = 0,
    min_symbol_obs: int = 60,
) -> Dict[str, float]:
    """每个品种上 corr(signal_t, fwd_ret_{t+lag})。

    lag=0 为标配：signal_t 与同行 fwd_close_ret_t（fwd_close_ret 已是 t→t+1 收益）。
    lag>0 表示信号对更远期收益的预测，用于衰减分析。
    """
    out: Dict[str, float] = {}
    for sym, gg in _per_symbol_panel(signal, panel, min_symbol_obs=min_symbol_obs):
        if lag != 0:
            gg = gg.copy()
            gg["ret"] = gg["ret"].shift(-lag)
            gg = gg.dropna()
            if len(gg) < min_symbol_obs:
                continue
        c = gg["sig"].corr(gg["ret"], method=method)
        if c == c:
            out[sym] = float(c)
    return out


def factor_ic_summary(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    method: str = "pearson",
    min_symbol_obs: int = 60,
) -> Dict[str, float]:
    """汇总每品种 IC：均值、标准差、IR、>0 比例、|IC| 均值。"""
    per_sym = per_symbol_ic(signal, panel, method=method, min_symbol_obs=min_symbol_obs)
    if not per_sym:
        return {
            "ic_mean": float("nan"),
            "ic_abs_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "ic_pos_frac": float("nan"),
            "n_symbols": 0,
        }
    arr = np.array(list(per_sym.values()), dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan")
    ir = float(mu / sd) if sd and sd == sd else float("nan")
    return {
        "ic_mean": mu,
        "ic_abs_mean": float(np.nanmean(np.abs(arr))),
        "ic_std": sd,
        "ic_ir": ir,
        "ic_pos_frac": float(np.mean(arr > 0)),
        "n_symbols": int(len(arr)),
    }


def per_symbol_ts_sharpe_no_demean(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    periods: int = 252,
    min_symbol_obs: int = 60,
) -> Dict[str, float]:
    """每品种纯 TS Sharpe：sig_t × fwd_ret_t 的年化 Sharpe，不做 CS-demean。"""
    out: Dict[str, float] = {}
    for sym, gg in _per_symbol_panel(signal, panel, min_symbol_obs=min_symbol_obs):
        pnl = gg["sig"] * gg["ret"]
        x = pnl.dropna()
        if len(x) < max(min_symbol_obs, periods // 4):
            continue
        mu, sd = x.mean(), x.std(ddof=1)
        if sd and sd == sd and sd != 0:
            out[sym] = float(np.sqrt(periods) * mu / sd)
    return out


def factor_ts_sharpe_summary(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    periods: int = 252,
    max_date=None,
    min_symbol_obs: int = 60,
) -> Dict[str, float]:
    p = panel
    if max_date is not None:
        md = pd.Timestamp(max_date)
        d = panel.index.get_level_values("date")
        p = panel.loc[d <= md]
    per_sym = per_symbol_ts_sharpe_no_demean(signal, p, periods=periods, min_symbol_obs=min_symbol_obs)
    if not per_sym:
        return {
            "ts_sharpe_mean": float("nan"),
            "ts_sharpe_median": float("nan"),
            "ts_sharpe_abs_mean": float("nan"),
            "ts_sharpe_pos_frac": float("nan"),
        }
    arr = np.array(list(per_sym.values()), dtype=float)
    return {
        "ts_sharpe_mean": float(np.nanmean(arr)),
        "ts_sharpe_median": float(np.nanmedian(arr)),
        "ts_sharpe_abs_mean": float(np.nanmean(np.abs(arr))),
        "ts_sharpe_pos_frac": float(np.mean(arr > 0)),
    }


def factor_t_stat(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_symbol_obs: int = 60,
) -> Dict[str, float]:
    """池化 t-stat：把所有 (date, symbol) 的 sig×ret 视为一个样本算 mean / SE。

    另给「按品种 t-stat 再取均值」的指标作为辅助。SE 假设 i.i.d.，故偏乐观；
    需要严格统计推断请改用 Newey–West / 块自助。
    """
    pooled: list[float] = []
    per_sym_t: list[float] = []
    for sym, gg in _per_symbol_panel(signal, panel, min_symbol_obs=min_symbol_obs):
        pnl = (gg["sig"] * gg["ret"]).dropna().values
        if len(pnl) < min_symbol_obs:
            continue
        pooled.extend(pnl.tolist())
        mu, sd = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
        if sd and sd == sd and sd != 0:
            per_sym_t.append(mu / (sd / np.sqrt(len(pnl))))
    if not pooled:
        return {"t_stat_pooled": float("nan"), "t_stat_per_symbol_mean": float("nan")}
    arr = np.array(pooled, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    n = len(arr)
    t_pool = mu / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    return {
        "t_stat_pooled": float(t_pool),
        "t_stat_per_symbol_mean": float(np.mean(per_sym_t)) if per_sym_t else float("nan"),
    }


def ic_decay(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    lags: List[int] = None,
    method: str = "pearson",
    min_symbol_obs: int = 60,
) -> Dict[int, float]:
    """IC 衰减曲线：lag>0 表示用 signal_t 预测 fwd_ret_{t+lag}。"""
    if lags is None:
        lags = [0, 1, 2, 3, 5, 10, 20]
    out: Dict[int, float] = {}
    for lg in lags:
        per_sym = per_symbol_ic(signal, panel, method=method, lag=lg, min_symbol_obs=min_symbol_obs)
        if per_sym:
            out[int(lg)] = float(np.nanmean(list(per_sym.values())))
        else:
            out[int(lg)] = float("nan")
    return out


def evaluate_factor(
    name: str,
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    decay_lags: List[int] = None,
    min_symbol_obs: int = 60,
) -> Dict[str, object]:
    """合并 IC（Pearson + Spearman）、TS Sharpe、t-stat、衰减 的单因子评价字典。"""
    ic_p = factor_ic_summary(signal, panel, method="pearson", min_symbol_obs=min_symbol_obs)
    ic_s = factor_ic_summary(signal, panel, method="spearman", min_symbol_obs=min_symbol_obs)
    ts = factor_ts_sharpe_summary(signal, panel, min_symbol_obs=min_symbol_obs)
    tt = factor_t_stat(signal, panel, min_symbol_obs=min_symbol_obs)
    decay = ic_decay(signal, panel, lags=decay_lags, method="pearson", min_symbol_obs=min_symbol_obs)
    t_panel = panel_signal_beta_cluster_date_t(signal, panel)
    t_hac_stack = panel_signal_beta_hac_stacked_t(signal, panel)
    t_dbc = panel_signal_beta_double_cluster_t(signal, panel)
    t_dk = panel_signal_beta_driscoll_kraay_t(signal, panel)
    return {
        "name": name,
        "ic_pearson": ic_p,
        "ic_spearman": ic_s,
        "ts_sharpe": ts,
        "t_stat": tt,
        "t_stat_panel_ols_cluster_date": t_panel,
        "t_stat_panel_hac_stacked": t_hac_stack,
        "t_stat_panel_double_cluster": t_dbc,
        "t_stat_panel_driscoll_kraay": t_dk,
        "ic_decay_pearson": {str(k): v for k, v in decay.items()},
    }


def evaluate_factor_oos(
    name: str,
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    train_end: pd.Timestamp | str,
    decay_lags: List[int] | None = None,
    min_symbol_obs: int = 30,
) -> Dict[str, object]:
    """仅 ``date > train_end`` 子样本上的 ``evaluate_factor``（用于样本外披露）。"""
    tr = pd.Timestamp(train_end)
    d = panel.index.get_level_values("date")
    panel_oos = panel.loc[d > tr]
    sig_oos = signal.loc[signal.index.get_level_values("date") > tr]
    return evaluate_factor(name, sig_oos, panel_oos, decay_lags=decay_lags, min_symbol_obs=min_symbol_obs)


def _stack_panel_yx(signal: pd.Series, panel: pd.DataFrame) -> pd.DataFrame | None:
    if signal.index.names != ["date", "symbol"]:
        return None
    df = pd.DataFrame({"y": panel["fwd_close_ret"], "x": signal}).dropna()
    return df if len(df) > 0 else None


def panel_signal_beta_hac_stacked_t(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_obs: int = 2500,
    maxlags: int = 5,
) -> float:
    """面板按 (date, symbol) 排序后 pooled OLS 的 signal 系数 t，HAC 协方差（拉直后 NW）。"""
    try:
        import statsmodels.api as sm
    except ImportError:
        return float("nan")
    df = _stack_panel_yx(signal, panel)
    if df is None or len(df) < min_obs:
        return float("nan")
    df = df.reset_index()
    df = df.sort_values(["date", "symbol"])
    y = df["y"].astype(float).values
    x = sm.add_constant(df["x"].astype(float).values)
    try:
        res = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return float(res.tvalues[1])
    except Exception:
        return float("nan")


def panel_signal_beta_double_cluster_t(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_obs: int = 2500,
) -> float:
    """linearmodels.PanelOLS：品种 × 时间双向聚类稳健 t（signal 系数）。"""
    df = _stack_panel_yx(signal, panel)
    if df is None or len(df) < min_obs:
        return float("nan")
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return float("nan")
    syms = df.index.get_level_values("symbol").astype(str)
    tms = df.index.get_level_values("date")
    d2 = pd.DataFrame(
        {
            "y": df["y"].astype(float).values,
            "const": 1.0,
            "x": df["x"].astype(float).values,
        },
        index=pd.MultiIndex.from_arrays([syms, tms], names=["symbol", "time"]),
    ).sort_index()
    try:
        mod = PanelOLS(d2[["y"]], d2[["const", "x"]], entity_effects=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        ts = res.tstats
        if hasattr(ts, "loc"):
            return float(ts.loc["x"])
        return float(ts.iloc[1])
    except Exception:
        return float("nan")


def panel_signal_beta_driscoll_kraay_t(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_obs: int = 2500,
    maxlags: int = 5,
) -> float:
    """PanelOLS + Bartlett kernel 协方差（Driscoll–Kraay 类面板稳健 t）。"""
    df = _stack_panel_yx(signal, panel)
    if df is None or len(df) < min_obs:
        return float("nan")
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return float("nan")
    syms = df.index.get_level_values("symbol").astype(str)
    tms = df.index.get_level_values("date")
    d2 = pd.DataFrame(
        {
            "y": df["y"].astype(float).values,
            "const": 1.0,
            "x": df["x"].astype(float).values,
        },
        index=pd.MultiIndex.from_arrays([syms, tms], names=["symbol", "time"]),
    ).sort_index()
    try:
        mod = PanelOLS(d2[["y"]], d2[["const", "x"]], entity_effects=False)
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=maxlags + 1)
        ts = res.tstats
        if hasattr(ts, "loc"):
            return float(ts.loc["x"])
        return float(ts.iloc[1])
    except Exception:
        return float("nan")


def hac_newey_west_t_mean(
    daily_series: pd.Series, *, maxlags: int = 5
) -> float:
    """对日度收益序列检验 E[r]=0：OLS 常数项 + Newey–West HAC 协方差（时间序列稳健）。"""
    try:
        import statsmodels.api as sm
    except ImportError:
        return float("nan")
    y = daily_series.dropna().astype(float)
    if len(y) < maxlags + 30:
        return float("nan")
    x = np.ones((len(y), 1))
    res = sm.OLS(y.values, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0])


def panel_signal_beta_cluster_date_t(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    min_obs: int = 2500,
) -> float:
    """长面板 pooled OLS：fwd_ret ~ const + signal，标准误按 **交易日** 聚类。

    用于缓解「同日多品种残差相关」导致的 pooled i.i.d. t 偏乐观问题；与纯截面 IC
    不同，此处仍是 (date,symbol) 层面的 TS 预测回归。
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return float("nan")
    if signal.index.names != ["date", "symbol"]:
        raise ValueError("signal must have MultiIndex [date, symbol]")
    df = pd.DataFrame({"y": panel["fwd_close_ret"], "x": signal}).dropna()
    if len(df) < min_obs:
        return float("nan")
    y = df["y"].astype(float).values
    x = sm.add_constant(df["x"].astype(float).values)
    dates = pd.factorize(df.index.get_level_values("date").astype(np.int64))[0]
    try:
        res = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": dates})
        return float(res.tvalues[1])
    except Exception:
        return float("nan")
