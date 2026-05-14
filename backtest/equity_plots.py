from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import pandas as pd

from mfe5210_cta.backtest.engine import (
    daily_turnover_from_signal,
    equity_and_underwater_series,
    net_returns_after_linear_cost,
    portfolio_returns_from_signal,
)


def _safe_png_stem(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _gross_net_daily(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    cross_section_demean: bool,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series | None]:
    gross = portfolio_returns_from_signal(signal, panel, cross_section_demean=cross_section_demean)
    if cost_bps <= 0:
        return gross, None
    turn = daily_turnover_from_signal(signal, cross_section_demean=cross_section_demean)
    net = net_returns_after_linear_cost(
        gross, turn, cost_bps_per_unit_turnover=cost_bps
    )
    return gross, net


def save_single_factor_equity_png(
    factor_name: str,
    gross_daily: pd.Series,
    net_daily: pd.Series | None,
    path: Path,
    *,
    cost_bps: float,
    cross_section_demean: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eq_g, uw_g = equity_and_underwater_series(gross_daily)

    fig, (ax_eq, ax_uw) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    cs = "CS-demean" if cross_section_demean else "no CS-demean"
    ax_eq.set_title(f"{factor_name} — single-factor portfolio ({cs})")
    ax_eq.plot(eq_g.index, eq_g.values, color="C0", linewidth=1.0, label="Gross")

    if net_daily is not None:
        eq_n, uw_n = equity_and_underwater_series(net_daily)
        ax_eq.plot(eq_n.index, eq_n.values, color="C1", linewidth=1.0, alpha=0.9, label=f"Net ({cost_bps:g} bps)")
        ax_uw.plot(uw_n.index, uw_n.values, color="C1", linewidth=0.9, alpha=0.8, linestyle="--", label="UW net")

    ax_uw.plot(uw_g.index, uw_g.values, color="C0", linewidth=0.9, label="UW gross")
    ax_eq.set_ylabel("Equity (cumprod)")
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_uw.set_ylabel("Drawdown")
    ax_uw.set_xlabel("Date")
    ax_uw.legend(loc="lower left", fontsize=7)
    ax_uw.grid(True, alpha=0.25)
    ax_eq.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_equity_overview_grid_png(
    series_by_factor: Mapping[str, pd.Series],
    path: Path,
    *,
    title: str,
) -> None:
    """各因子 Gross 累计净值在同一图中归一化为起点 1（仅视觉对比）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(series_by_factor.keys())
    if not names:
        return
    n = len(names)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), squeeze=False)
    for i, name in enumerate(names):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        gross = series_by_factor[name]
        eq, _ = equity_and_underwater_series(gross)
        if len(eq):
            eq0 = float(eq.iloc[0])
            y = eq / eq0 if eq0 != 0 else eq
            ax.plot(y.index, y.values, color="C0", linewidth=0.8)
        ax.set_title(name, fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.2)
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_all_factor_equity_plots(
    signals: Mapping[str, pd.Series],
    panel: pd.DataFrame,
    output_dir: Path,
    *,
    cross_section_demean: bool,
    cost_bps: float = 0.0,
    only_factors: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """
    为每个因子写一张 cumulative equity + underwater 图；另写 equity_overview_grid.png。
    返回因子名 → 单图路径。
    """
    output_dir = Path(output_dir)
    out: Dict[str, Path] = {}
    wanted = set(only_factors) if only_factors is not None else None
    gross_map: MutableMapping[str, pd.Series] = {}

    for name, sig in signals.items():
        if wanted is not None and name not in wanted:
            continue
        gross, net = _gross_net_daily(
            sig, panel, cross_section_demean=cross_section_demean, cost_bps=cost_bps
        )
        gross_map[name] = gross
        stem = _safe_png_stem(name)
        p = output_dir / f"{stem}.png"
        save_single_factor_equity_png(
            name,
            gross,
            net,
            p,
            cost_bps=cost_bps,
            cross_section_demean=cross_section_demean,
        )
        out[name] = p

    cs = "CS-demean on" if cross_section_demean else "CS-demean off"
    cost_note = f", net @ {cost_bps:g} bps" if cost_bps > 0 else ""
    save_equity_overview_grid_png(
        gross_map,
        output_dir / "equity_overview_grid.png",
        title=f"Gross equity (normalized to 1 at first obs) — {cs}{cost_note}",
    )
    return out
