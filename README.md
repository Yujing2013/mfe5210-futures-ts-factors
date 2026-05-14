# MFE5210 — Time-Series CTA Alpha on Chinese Commodity Futures

This project meets the course **Alpha Factors Development** submission requirements: assemble data, publish factor-generation code on GitHub with a **README** (correlation matrix, no-cost average Sharpe, exposition), plus references.

## Key numbers (`outputs/summary.json`)

> Open **`primary_metrics`**. Factor count varies with columns available in your panel (this public repo loads **only a local parquet**; optional columns activate extra factor blocks).

| Metric | JSON field | Meaning |
|--------|-----------|---------|
| Constraint **≤ 0.5** (assignment-style) | `max_abs_pnl_residual_corr_selected_in_sample` | Greedy subset uses residual correlation constraint **inside the train window only** (`--sign-train-frac` cutoff); **no OOS lookahead** |
| Full-sample diagnostics | `max_abs_pnl_residual_corr_selected_full_sample` | Worst pairwise residual correlation among the selected set on **all** dates (**may exceed 0.5**; normal OOS) |
| Pure OOS residual corr | `max_abs_pnl_residual_corr_selected_oos` / `oos_metrics.*` | Same selected names, correlations computed only on \( t >\) train end |
| Portfolio Sharpe | `portfolio_no_demean_mean_sharpe_selected` | Equal-weight TS portfolio, **no CS-demean** |
| Robust t (portfolio returns) | `mean_t_stat_hac_portfolio_selected`, `per_factor.*.t_stat_hac_portfolio_returns` | Newey–West HAC (lag=5) |
| Robust t (panel OLS stack) | `mean_t_stat_panel_cluster_date_selected`, etc. | Pooled `fwd_ret ~ signal`: date clustered SE, stacked HAC, two-way DK-style estimators |
| Legacy supplementary | `supplementary.*` | CS-demean Sharpe branches, signal-similarity matrix |

**Greedy rule**: sort by IS `ts_sharpe_mean` then sequentially add factors with pairwise residual corr \(|ρ| ≤ 0.5\) in sample; ordering field: `primary_metrics.ordering_for_greedy`.

## Why we residualize PnL for correlation

Naïve factor PnL is usually highly correlated with the equal-weight market return \( R^{\mathrm{mkt}}_t \). Pairwise correlations of gross PnL can sit around 0.8+ **even when signals differ**.

We regress daily factor PnL on \( R^{\mathrm{mkt}}_t \),

\[
\mathrm{PnL}_t \;=\; \alpha + \beta\, R^{\mathrm{mkt}}_t + \varepsilon_t, \quad \hat\beta = \mathrm{cov}(\mathrm{PnL}, R^{\mathrm{mkt}})/\mathrm{var}(R^{\mathrm{mkt}})
\]

Then take **Pearson correlation of residuals \(\varepsilon\)** — interpreted as correlation of **factor alpha orthogonal to commodity beta**. Implementation: [`metrics.pnl_residual_correlation_matrix`](backtest/metrics.py).

Plots: [`outputs/corr_matrix_pnl_residual_all.png`](outputs/corr_matrix_pnl_residual_all.png), [`outputs/corr_matrix_pnl_residual_selected.png`](outputs/corr_matrix_pnl_residual_selected.png).

Supplementary signal similarity: [`outputs/corr_matrix_signal_similarity.png`](outputs/corr_matrix_signal_similarity.png).

## Factor evaluation primitives

See [`backtest/ts_evaluation.py`](backtest/ts_evaluation.py):

- **IC**: Within each symbol \(\mathrm{corr}(\mathrm{signal}_t,\mathrm{fwd}_t)\), Pearson/Spearman, cross-sectional summaries.
- **TS Sharpe**: Per symbol \(\sqrt{252}\cdot \mathrm{mean}(\mathrm{sig}\cdot r)/\mathrm{std}\), **without** CS-demean; averaged across symbols.
- **Panel OLS \(t\)**: clustered, HAC, two-way clustered, DK-style covariance families as reported in `summary.json`.

IC decay PNGs live under **`outputs/ic_decay/`** (`--ic-decay-plots` on by default).

## Sign calibration (no lookahead)

Many TS motifs (momentum vs reversal) invert on Chinese commodities. We flip each factor \(\pm 1\) using **portfolio Sharpe only on the first `train_frac` of calendar dates** (`--sign-train-frac`, default `0.5`), then freeze the sign thereafter. Detail: `summary.json.sign_calibration` including `oos_sharpe_signed`. Disable via `--no-sign-calibration`.

## Environment (public repo: **local parquet only**)

```bash
cd /path/to/parent/of/mfe5210_cta
pip install -r mfe5210_cta/requirements.txt
```

| Default path | MultiIndex `(date, symbol)`; minimum columns typically `open/high/low/close`, `volume`, `hold`; optional extras below |
|----------------|--------------------------------------------------------------------------------------------------------|
| `mfe5210_cta/data/cache/panel.parquet` | |

```bash
python3 -m mfe5210_cta.run_pipeline
python3 -m mfe5210_cta.run_pipeline --cache /path/to/your_panel.parquet
```

- **`--force-download`**: Does **not** contact the internet here; if the parquet exists it only forces **re-reading** disk.
- This repository **excludes** third-party downloader code (`mfe5210_cta/.gitignore`). Use any compliant panel you construct offline.

## Data columns

[`data/download.py`](data/download.py) **reads parquet only**. New runs set `summary.json` → **`data_source` = `"local_parquet"`** (historic JSON in `outputs/` may still say `"tushare"` if not regenerated).

**Required / common**: `open`, `high`, `low`, `close`, `volume`, `hold`. **Optional extras** (unlock extra registrations): `settle`, `amount`, `basis_dom_pct`, `term_spread_pct`, `member_net_long_top20`, `wsr_vol`, `index_close`.

Universe anchor list: **`FUTURES_MAIN_CONT_SYMBOLS`** in [`data/config.py`](data/config.py). You may use different symbol strings in your panel; load path returns whatever rows exist.

## Out-of-sample (OOS)

- `oos_metrics`, `per_factor_oos`: same greedy selection as IS but statistics on \( t>\) greedy train cutoff.

## Factors (time-series)

Each raw factor applies **within symbol** \(\rightarrow\) rolling z-score (\(60\) bars) \(\rightarrow\) rolling cross-section-Free winsor \(\rightarrow\) \(\pm\) sign calibration. Registry: [`factors/registry.py`](factors/registry.py).

**Baseline ~27 labels** \(+\) conditional blocks if `settle` / `amount` / extension columns populated.

Full formulas and references: [`FACTORS.md`](FACTORS.md).

### Backtesting & attribution (research, no commissions in primary block)

\[ \mathrm{pnl}_{s,t}=\mathrm{signal}_{s,t}\times \mathrm{fwd\_close\_ret}_{s,t},\quad \mathrm{fwd}=\frac{C_{t+1}}{C_t}-1. \]

Equal-weight averaging across symbols, **without** CS-demean in the headline Sharpe definitions.

Residual-corr greedy selection thresholds \(|ρ|≤0.5\) use **PnL residuals vs market**.

Equity PNGs (\(`--equity-plots`\), default on) and `--cost-bps` overlay described in argparse help.

### `summary.json` top-level map

| Field | Description |
|---------|-------------|
| `data_source` | **`local_parquet`** on fresh runs (older committed JSON may still read `tushare`) |
| `tushare_skipped` | Legacy list name — often empty for local-only caches |
| `primary_metrics`, `per_factor`, `oos_metrics`, `supplementary`, `sign_calibration`, `methodology_notes` | As generated |
| `per_symbol_ts_sharpe` | Also exported CSV `outputs/per_symbol_ts_sharpe.csv` |

## Leakage checklist

1. \(\mathrm{pnl}=\mathrm{sig}\times \mathrm{fwd\_close\_ret}\) **without extra** `shift(-1)` on returns.
2. Feature construction sticks to **lags / rolls up to \(t\)** for same-row `fwd`.
3. Rolling z-score and winsor rely only on \(\le t\) history per symbol.
4. Winsor clipping is strictly **within symbol**, never cross-section rank on a date.

Residual roll/spread semantics follow **your** stitched continuous contract conventions.

## Repository layout

| Path | Purpose |
|------|---------|
| `mfe5210_cta/data/` | Config + parquet I/O helpers |
| `mfe5210_cta/factors/` | Definitions + registry + transforms |
| `mfe5210_cta/backtest/` | Engines, diagnostics, plotting |
| `mfe5210_cta/run_pipeline.py` | One-shot compute + artefacts |
| `mfe5210_cta/outputs/` | Figures & JSON (**commit for grading evidence**) |

## References

1. Jegadeesh & Titman (1993) – Momentum classic.
2. Moskowitz, Ooi & Pedersen (2012) – Time-series momentum.
3. Garman & Klass (1980); Parkinson (1980) – Range volatility.
4. Amihud (2002) – Illiquidity.
5. Lehmann (1990) – Short reversal.
6. Baltas & Kosowski (2013) – Vol-scaled trend.
7. Bollen & Whaley (2004) – Open interest motifs.
8. Williams (1979) – `%R`-style oscillator.
9. Hong & Yogo (2012) – Commodity OI hypotheses.
10. Course reading list & documentation for any commercial data feeds you privately use (**not bundled here**).

## Research vs brokerage-grade realism

Baseline outputs summarize **research-scale** exposures (signals × stitched contract returns **without explicit margin/leverage bookkeeping**). If you extrapolate dollar PnL, add multipliers, lot constraints, commissions, rolls, latency. Compare factors **under identical plumbing** stays valid regardless.

---

**Disclaimer**: coursework only; **not** investment advice. Default curves are analytical composites—see caveat above before citing dollar performance.
