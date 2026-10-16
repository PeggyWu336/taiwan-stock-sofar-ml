# Taiwan Stock Return Prediction: SOFAR + Machine Learning

Combining sparse factor extraction (SOFAR) with ElasticNet / Random Forest / XGBoost
to predict monthly stock returns in the Taiwan equity market (2005-2025).
Master's thesis research, National Chengchi University (NCCU), Dept. of Economics.

## Key Results

Out-of-sample period: Jan 2016 - Nov 2025 (119 months). All figures are for a
decile long-short portfolio, net of transaction costs (0.6% round trip).

| Metric | Value | What it means |
|---|---|---|
| Sharpe Ratio (XGBoost-Hybrid) | **1.425** | Risk-adjusted return: how much return per unit of risk taken |
| Newey-West t-stat | **5.541** | Statistical significance: is this skill or luck? (>2 is considered significant) |
| Spanning Test alpha | 0.00329 (t = 3.498, p < 1%) | About 0.3% excess return per month that cannot be replicated by the Market-only and SOFAR-only strategies combined |
| Feature importance (Gain) | Market 44.6% / SOFAR 55.4% | Both signal types contribute roughly equally, so SOFAR factors are not decorative |
| Consistency | 9 of 10 years with positive Sharpe | The only negative year is 2025 (tariff shock) |

**Comparison with a passive benchmark** (Yuanta Taiwan Top 50 ETF, ticker 0050, same period):

| | XGBoost-Hybrid (long-short) | 0050 ETF (buy and hold) |
|---|---|---|
| Annualized return | 13.61% | 18.02% |
| Sharpe Ratio | **1.425** | 1.051 |
| Max drawdown | **-8.77%** | -24.29% |

The strategy earns a lower raw return than 0050 but with much better risk-adjusted
performance and far smaller drawdowns. Note that a long-short portfolio and a
long-only ETF are different instruments; 0050 figures are before costs
(negligible for a passive ETF).

See `results/thesis_output/` for full tables and figures.

## The Idea

Taiwan stock returns are noisy, and quarterly financial statements contain many
overlapping numbers (ROA vs. ROE, gross margin vs. operating margin, ...).
Feeding all 25 raw variables into a model is like handing it a messy closet.

**SOFAR (Sparse Orthogonal Factor Regression)** tidies the closet into a few drawers:
each drawer (factor) is built from only a handful of related variables (sparse),
and no two drawers overlap (orthogonal). Compared with PCA, where every drawer
contains a bit of everything, the factors are easier to interpret and less noisy.
(In the thesis, SOFAR versions beat the PCA versions on return, Sharpe and drawdown.)

The research question: **do these cleaned-up fundamental factors carry information
that market signals (momentum, volatility, liquidity) do not?**
To test it, three feature sets are compared across three algorithms (3 x 3 = 9 models),
plus an OLS-3 linear benchmark (size, value, 12-month momentum):

| Feature set | Contents |
|---|---|
| Market-only | Momentum, volatility, liquidity |
| SOFAR-only | Factors extracted from 25 quarterly financial variables |
| Hybrid | Both combined |

## How It Works

```mermaid
flowchart LR
    A["TEJ data, 2005-2025"] --> B["Monthly universe filter<br/>about 940 stocks per month"]
    B --> C["25 quarterly financial variables"]
    B --> D["Price and volume signals<br/>momentum, volatility, liquidity"]
    C --> E["SOFAR<br/>sparse factors"]
    D --> G["Features"]
    E --> G
    G --> H["ElasticNet / Random Forest / XGBoost<br/>expanding window, refit every year"]
    H --> I["Predicted next-month return"]
    I --> J["Decile long-short portfolio<br/>buy top 10%, short bottom 10%"]
    J --> K["Net of costs: Sharpe, Newey-West t, spanning test"]
```

Design choices that guard against look-ahead bias and overfitting:

- **Expanding window, annual refit**: each year's model is trained only on data up to the previous December, then predicts the next 12 months.
- **Reporting lag**: quarterly financials are only used after their latest legal disclosure date (e.g. Q1 reports from May), so the model never sees numbers investors did not yet have.
- **Liquidity filter**: each month keeps only stocks above the 30th percentile of both market cap and trading value, at least 30 stocks.
- **Transaction costs**: 0.6% round trip charged on portfolio turnover.
- **Fixed hyperparameters and seed (42)**: set in advance, not tuned on out-of-sample results.

## Repository Contents

| Folder | Content |
|---|---|
| `src/` | R code: full pipeline, thesis tables and figures, comparison with 0050, monthly returns for Appendix D.1 |
| `results/thesis_output/` | Thesis tables (CSV, Tables 4-11, A.1, B.1) and figures (PNG, Figures 1-2) |
| `data/` | Data documentation only (raw TEJ data is licensed and not included) |

Where to look first:

| File | Content |
|---|---|
| `src/pipeline_main.R` | End-to-end pipeline: universe filter, imputation, SOFAR, ML models, portfolios |
| `results/thesis_output/Table7_portfolio_perf.csv` | Portfolio performance of all 10 models |
| `results/thesis_output/Table8_spanning_test.csv` | Spanning test (information complementarity) |
| `results/thesis_output/Figure2_cumret_wealth.png` | Cumulative wealth of the main models |

## Data

- **Source:** Taiwan Economic Journal (TEJ), 2005-01 to 2025-12, monthly
- **Sample:** Taiwan listed and OTC common stocks (TDRs excluded, all industries kept)
- **Universe:** re-screened monthly; market cap and trading value must both be
  in the top 70%, with at least 30 stocks per month (668 to 1,174 stocks per month)
- **Fundamental inputs:** 25 quarterly financial statement variables in 7 groups,
  lagged to official reporting deadlines to avoid look-ahead bias
- **Market inputs:** Mom1M, Mom6M and Mom12M (skip-1), 12-month return volatility,
  and liquidity (log average trading value)

TEJ is a licensed database, so the raw data are not included.
See [data/README.md](data/README.md) for the full input format.

## How to Run

### 1. Install R packages

```r
install.packages(c("data.table", "lubridate", "zoo", "rrpack", "glmnet",
                   "ranger", "xgboost", "sandwich", "lmtest", "ggplot2"))
```

### 2. Prepare the data

Put `monthly_panel_final.csv` and `0050.csv` in the `data/` folder.

### 3. Run the scripts in order

All scripts use relative paths, so run them **from the repository root**
(for example `Rscript src/pipeline_main.R`, or set the RStudio working
directory to the repo root).

| Step | Script | What it does |
|---|---|---|
| 1 | `src/pipeline_main.R` | Universe screening, feature engineering, yearly SOFAR factor extraction, training of all 10 models with an expanding window, and long-short backtest. Results go to `outputs/pipeline_runs/` |
| 2 | `src/output_tables_figures.R` | Builds the thesis tables (Table 4-11, A1, B1) and Figures 1-2 into a `thesis_output/` folder under `outputs/` |
| 3 | `src/backtest_vs_0050.R` | Compares XGBoost-Hybrid with the 0050 ETF over the same out-of-sample period |
| 4 | `src/three_year_annualized_return.R` | XGBoost-Hybrid returns for three selected years (2020, 2021, 2025), including the monthly long-short net returns in Appendix D.1. Reads the latest run automatically |

### Notes

- The full pipeline takes about **5-6 hours** on a typical laptop.
- Random seeds are fixed (`set.seed(42)`), so results are reproducible.
- `outputs/` is ignored by git. To update the published results, copy the
  files from the generated `thesis_output/` folder into `results/thesis_output/`.

## Limitations

- **Nonlinear models broke down in 2025.** All three XGBoost models had a
  negative Sharpe ratio in 2025 (XGBoost-Hybrid: -0.411), while the linear
  OLS-3 (0.662) and the pure SOFAR factor ranking (1.266) stayed positive.
  Losses were spread across the year after the April 2025 tariff shock,
  suggesting the patterns learned from history stopped working under a
  structural break. A regime-aware switch to simpler models is left for
  future work.
- **SOFAR factors are hard to interpret over time.** Factors are re-estimated
  every year, so a factor with the same number can load on different
  variables in different years. The analysis therefore focuses on the total
  contribution of SOFAR factors, not on the meaning of each one.
- **Factor count hits the upper bound.** The ED method selected the maximum
  of 12 factors in every year, likely because cross-sectional Z-scoring
  compresses eigenvalue gaps. Choosing the count by out-of-sample error or
  cross-validation would be a better alternative.
- **Simplified trading costs.** Costs are a flat 0.6% round trip. Borrowing
  fees, short-sale constraints and market impact are not modeled, so the
  long-short returns are likely an upper bound of what is achievable.
- **Not a like-for-like comparison with 0050.** 0050 had a higher raw annual
  return (18.02% vs 13.61%). The strategy wins on risk-adjusted terms
  (Sharpe 1.425 vs 1.051, MDD -8.77% vs -24.29%), but it is a dollar-neutral
  long-short portfolio, while 0050 is a long-only index fund.
