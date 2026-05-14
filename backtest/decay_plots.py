"""IC 衰减可视化：单图 = 单因子的 lag → mean IC（跨品种平均）柱状图；
另出一张 27 因子总览网格图。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfe5210_cta.backtest.ts_evaluation import ic_decay


DEFAULT_LAGS = [0, 1, 2, 3, 5, 10, 20]


def _plot_decay_bars(
    name: str, decay: Dict[int, float], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    lags = sorted(decay.keys())
    vals = [decay[k] for k in lags]
    colors = ["#2c7fb8" if v >= 0 else "#cb181d" for v in vals]
    ax.bar([str(l) for l in lags], vals, color=colors)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("mean IC (Pearson) across symbols")
    ax.set_title(f"IC decay — {name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_decay_overview(
    decays: Dict[str, Dict[int, float]], path: Path, lags: List[int]
) -> None:
    names = list(decays.keys())
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.0), sharey=True)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, name in zip(axes_flat, names):
        d = decays[name]
        vals = [d.get(l, np.nan) for l in lags]
        colors = ["#2c7fb8" if (v == v and v >= 0) else "#cb181d" for v in vals]
        ax.bar([str(l) for l in lags], vals, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(name, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes_flat[len(names):]:
        ax.set_axis_off()
    fig.suptitle(
        f"IC decay — lags {lags} (Pearson, per-symbol corr averaged)", fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_ic_decay_plots(
    signals: Dict[str, pd.Series],
    panel: pd.DataFrame,
    output_dir: Path,
    *,
    lags: List[int] = None,
) -> Dict[str, Dict[int, float]]:
    """对每个因子绘 IC 衰减柱图 + 一张总览网格，返回各因子的衰减字典。"""
    if lags is None:
        lags = DEFAULT_LAGS
    output_dir.mkdir(parents=True, exist_ok=True)
    decays: Dict[str, Dict[int, float]] = {}
    for name, sig in signals.items():
        d = ic_decay(sig, panel, lags=lags, method="pearson")
        decays[name] = d
        _plot_decay_bars(name, d, output_dir / f"{name}.png")
    _plot_decay_overview(decays, output_dir / "ic_decay_overview.png", lags)
    return decays
