# Sales Demand Forecasting & Anomaly Detection

An end-to-end Data Science project built on the **[Corporación Favorita "Store Sales"
dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)** (Kaggle),
taken from a raw CSV to an interactive dashboard, with a `reason-over-complexity` philosophy
throughout.

- **What**: forecast daily network-level (and store-type / top-store) sales, then flag
  days whose sales *deviate from the expected pattern*.
- **Data**: 1,017,209 store-day rows · 1,115 stores · 2013-01-01 → 2015-07-31 (942 days).
- **Analysis level**: *store / store-type / promo-period* — not product-level.
- **Result (honest, out-of-sample)**: the final model cuts forecasting error ~84% vs a
  strong seasonal-naive baseline.

---

## Project phases (all complete)

| # | Phase | Key deliverable |
|---|-------|-----------------|
| 1 | Data understanding | `notebooks/01_data_understanding.ipynb` |
| 2 | Cleaning & EDA | `notebooks/02_eda.ipynb` + cleaned CSVs |
| 3 | Feature engineering (35 features, leakage-verified) | `notebooks/03_feature_engineering.ipynb` |
| 4 | Baselines (naive / moving average; seasonal naive wins) | `notebooks/04_baseline_forecasting.ipynb` |
| 5 | Final forecast model (SARIMA vs ML → **HistGradientBoosting wins**) | `notebooks/05_final_forecast_model.ipynb` |
| 6 | Anomaly detection (rolling Z-score; 121/942 days flagged at Z>2) | `notebooks/06_anomaly_detection.ipynb` |
| 7 | SQL analysis (SQLite: 30+ business queries) | `notebooks/07_sql_analysis.ipynb` + `sql/analysis.sql` |
| 8 | Interactive Streamlit dashboard (5 pages) | `app/dashboard.py` |
| 9 | Testing, cleanup & this documentation | `tests/verify_project.py`, `README.md`, etc. |

---

## Repo layout

```
sales-demand-forecasting/
├── app/
│   └── dashboard.py          # Streamlit app (5 pages)
├── data/
│   ├── raw/                  # Kaggle download (git-ignored; ~38 MB train.csv)
│   └── processed/            # Cleaned CSVs, SQLite db, saved model  (auto-generated)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_forecasting.ipynb
│   ├── 05_final_forecast_model.ipynb
│   ├── 06_anomaly_detection.ipynb
│   └── 07_sql_analysis.ipynb
├── sql/
│   └── analysis.sql          # Every business query, runnable on data/processed/sales.db
├── tests/
│   └── verify_project.py     # 10 end-to-end checks (run: python tests/verify_project.py)
├── requirements.txt
├── README.md
├── RESUME_BULLETS.md         # Quantified, interview-ready bullet points
└── INTERVIEW_PREP.md         # Likely interview Q&A for every phase
```

> **Why no `src/`?** By design. Analysis lives in notebooks (beginner-friendly and
> auditable); cleaned CSVs carry state between phases; the only standalone code is the
> dashboard. Centralizing into a package would add abstraction without adding value for
> a first resume project — the project's governing constraint is understanding first.

---

## Getting started

Requires **Python 3.12+** (tested on 3.14). Install dependencies:

```bash
pip install -r requirements.txt
```

### 1. Run the notebooks (in order)

Notebooks **assert their own folder**: open them with Jupyter from inside `notebooks/`,
or execute headlessly from `notebooks/`:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_data_understanding.ipynb --output 01_data_understanding.ipynb
# ... repeat for 02 … 07
```

Each phase regenerates its outputs in `data/processed/` and hands state to the next.

### 2. Run the tests

```bash
python tests/verify_project.py
```

10 checks: project tree, notebook execution, data integrity, split boundaries,
**no-leakage** on lag/rolling features, model prediction, anomaly-table consistency,
SQLite answers, and a dashboard smoke test. Expect `ALL CHECKS PASSED`.

### 3. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Pages: **Overview** (KPIs + seasonality) · **Store types** · **Forecasting** (honest
out-of-sample + feature importances) · **Anomalies** · **SQL insights**.

---

## Headline findings

1. **Growth story is productivity, not revenue.** Network sales fell −5.3% 2013→2014,
   but sales per *open* store-day rose every year (+2.6% same-store) — the drop was
   *capacity* (fewer open days), not performance.
2. **Strong weekly seasonality.** Weekends carry the network; a lag-7 seasonal-naive
   baseline beats time-window averages and yesterday-naive, confirming the weekly cycle.
3. **The ML model wins big.** HistGradientBoosting on 32 features (lags, rolling stats,
   promo, calendar, interactions) reaches **validation MAE $0.335M / test MAE $0.283M**,
   versus $2.01M for seasonal naive. Top features: `sales_lag_7`, `rolling_mean_7`,
   `promo_share`, `is_weekend`, `sales_lag_1`.
4. **SARIMA works but is feature-blind.** `SARIMA(1,0,1)(1,1,1,7)` beats the baseline
   ($0.649M val) but cannot see promo/holiday inputs — the ML model's edge.
5. **Promotions lift sales ~39%** on average (correlational, not causal), and different
   store types respond differently.
6. **121 of 942 days** were flagged as anomalies at Z>2 — deviations from the model's
   expectation worth a human look (payday, holidays, spikes).

---

## Key methodology decisions (do not silently reverse)

- **Chronological** train / validation / test split (no random split, no R²): train
  2013-01-01→2015-06-30, validation 2015-07-01→15, test 2015-07-16→31.
- **One-day-ahead forecast discipline**: every feature is computed from information
  known the day *before*; lag/rolling features are recomputed from actuals only up to
  each prediction date. The Phase-3 leakage check (4 assertions) enforces this.
- **`Customers` is not a forecasting feature** (future values are unknowable) — historical
  insight only.
- **Closed days (Open=0) are "no sales opportunity", not zero demand**; they are kept,
  never imputed.
- **Anomalies = "differs from expected pattern"**, not errors or fraud. Z-score baseline
  on model residuals, compared against Isolation Forest in notebook 06.
- **Forecast scope**: network + 4 store types + top-5 case-study stores. No per-store
  model for 1,115 stores.
- Metrics: **MAE / RMSE / MAPE** (not R²).
- `test.csv` / `sample_submission.csv` in `data/raw/` are unused Kaggle scaffolding.

---

## How to extend (suggestions)

- **Rolling origin / backtesting**: evaluate with expanding-window origins for more
  robust error bars than a single 15/16-day holdout.
- **Exogenous features into SARIMAX**: the ML edge came from promo/holiday inputs —
  SARIMAX could recover part of it while staying a true time-series model.
- **Store clustering**: train one model per cluster of similar stores instead of the
  whole network.
- **Add `Customers` forecasts** as a two-stage model and use it as a feature *carefully*.
- **Calendar events**: fold in dated holiday lists (e.g., `holidays` package) instead of
  generic `state_holiday` flags.

---

## Notes for the reviewer / interviewer

See **`INTERVIEW_PREP.md`** — it walks through the "why" of every decision, likely
interview questions, and the limitations of each approach. **`RESUME_BULLETS.md`**
contains quantified, honest bullet points ready to drop into a resume.