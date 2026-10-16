# Data

The raw data used in this project are **not included** in this repository.
This page describes what the data look like, so the pipeline can be reproduced
by anyone with access to the same source.

## Source

- Database: Taiwan Economic Journal (TEJ), retrieved via TEJ Smart Wizard
- Sample period: 2005-01 to 2025-12 (252 months), monthly frequency
- Out-of-sample backtest: 2016-01 to 2025-11 (119 months)

## Why the data are not public

TEJ is a commercial, licensed database. Its terms do not allow redistribution
of raw or derived stock-level data. This repository therefore shares only the
code and the aggregated thesis results in `results/thesis_output/`.

## Sample and universe

- Taiwan listed (TWSE) and OTC (TPEx) common stocks
- TDRs excluded; all industries kept, including financials
- The universe is re-screened every month inside `src/pipeline_main.R`:
  - month-end market cap percentile >= 30%
  - average daily trading value percentile >= 30%
  - both conditions must hold; at least 30 stocks per month
- Result: 668 to 1,174 stocks per month (median 941),
  233,560 stock-month observations in total

## Required input files

Place the following files in `data/` before running the scripts.
CSV files in this folder are ignored by git (see `.gitignore`).

### 1. `data/monthly_panel_final.csv`

One row per stock per month.

| Column | Description |
|---|---|
| `stock_id` | Stock code |
| `ym` | Year-month |
| `monthly_ret` | Monthly return = (last close of month / first close of month) - 1, using dividend-adjusted prices. Set to missing if the first close is not above NT$1 |
| `mktcap_eom` | Market cap on the last trading day of the month |
| `avg_turnover_m` | Average daily trading value within the month |
| `pb_tej` | Price-to-book ratio (inverted to book-to-market for OLS-3) |
| `A_roa` ... `G_ebt` | 25 quarterly financial statement variables (see below) |

**Financial statement variables (25 in 7 groups)**

| Group | Count | Variables |
|---|---|---|
| A Profitability | 5 | ROA, ROE, gross margin, operating margin, net margin |
| B Growth | 5 | Revenue growth, EPS, gross profit growth, operating income growth, net income growth |
| C Cash flow | 2 | Operating cash flow, free cash flow |
| D Liquidity and leverage | 4 | Current ratio, quick ratio, debt ratio, interest coverage |
| E Efficiency | 3 | Inventory turnover, receivables turnover, total asset turnover |
| F Size | 3 | Total assets, parent equity, current assets |
| G Cash-flow structure | 3 | Investing cash flow, financing cash flow, pre-tax income |

Quarterly reports are mapped to months with a fixed reporting lag to avoid
look-ahead bias: Q1 is usable from May, Q2 from August, Q3 from November,
and Q4 from March of the following year. Each value is carried forward
until the next report becomes available.

### 2. `data/0050.csv`

Daily prices of the Yuanta Taiwan Top 50 ETF (0050), used only by
`src/backtest_vs_0050.R` as a passive benchmark.

| Column | Description |
|---|---|
| `year` | Trading date (Y/m/d) |
| `close` | Closing price |