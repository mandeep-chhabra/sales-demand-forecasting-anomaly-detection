# PROJECT_STATUS.md

> **Handoff document** — created 2026-09-10 after Phase 2. Read this before doing any
> work in the project. It tells you exactly where the project stands, what decisions
> were made and why, and what the next session must do.

---

## 1. Project objective

Build a **fresher-level Data Science resume project**: an end-to-end **Sales Demand
Forecasting & Anomaly Detection System** that

- analyzes historical daily store sales,
- forecasts future demand (network + store-type + top-store levels),
- detects unusual sales behavior (spikes/drops vs. expected patterns),
- and converts results into **quantified business insights**,
- exposed through a **simple Streamlit dashboard**.

Governing constraint (from the user's brief): **understanding > complexity**,
**good methodology > number of technologies**, **working project > fancy architecture**.
No Docker/Kafka/Spark/AWS etc. Keep it beginner-friendly and explainable.

Dataset decision (user-confirmed): **Corporación Favorita**, Kaggle "Store Sales –
Time Series Forecasting". Store-level (not product-level) — the analysis unit is
**stores / store types / promo periods** instead of products.
Be
## 2. Phase plan (all 9)

| # | Phase | Status |
|---|-------|--------|
| 1 | Dataset selection & data understanding | ✅ complete |
| 2 | Data cleaning & EDA | ✅ complete |
| 3 | Feature engineering | ✅ complete |
| 4 | Baseline forecast (naive / moving average) | ✅ complete |
| 5 | Final forecasting model (SARIMA vs tree-based ML) | ⏭️ NEXT |
| 5 | Final forecasting model (SARIMA vs tree-based ML) | pending |
| 6 | Anomaly detection (rolling mean/std + Z-score; compare Isolation Forest) | pending |
| 7 | SQL analysis (SQLite) | pending |
| 8 | Streamlit dashboard | pending |
| 9 | Testing, cleanup, documentation (README, resume bullets, interview prep) | pending |

**Working agreement:** complete one phase, explain it (what / why / implementation /
important code / how to test), then **wait for user confirmation** before the next.

## 3. Phase 1 — completed (data understanding)

Deliverable: `notebooks/01_data_understanding.ipynb` (~24 cells, executed).

- Confirmed the existing Kaggle Favorita dataset in `data/raw/`: **1,017,209 rows**,
  one row per store × day, **2013-01-01 → 2015-07-31** (942 days, ~31 months).
- Verified structure: 9 train columns, 1,115 stores, store metadata (10 cols) in `store.csv`.
- **No missing values in train**; missing metadata in `store.csv` (competition distance 0.27%,
  competition-since 31.7%, promo2 fields 48.8%).
- **No duplicates**, no repeated `(Store, Date)` pairs.
- **~38% of store-days are closed** (Open=0, Sales=0) — zero sales ≠ zero demand.
- Sales distribution heavily right-skewed; max $41,551; quantile check done.
- **33,121 absent store-days** flagged for investigation.
- **Mixed-type `StateHoliday`: string `'0'` (855K rows) and integer `0` (131K rows) coexist** —
  pandas threw a `DtypeWarning`; this became the top cleaning target.
- Assumption noted: ~2.5 yearly cycles → weekly seasonality expected strong, annual weak.

## 4. Phase 2 — completed (cleaning & EDA)

Deliverable: `notebooks/02_eda.ipynb` (32 cells, 8 figures, executed) +
**`data/processed/train_clean.csv` (36 MB)** and `data/processed/store_clean.csv`.

### Cleaning decisions (all traceable to Phase-1 findings)

| Action | Rationale |
|---|---|
| `StateHoliday`: `astype(str)` + strip + map `'0'→'none'` | Collapses the mixed-type `'0'`/`0` split; verified single group (986,159 "none") |
| `DayOfWeek` vs calendar: **0 mismatches** matrix-wide | Column trustworthy; kept as a feature |
| `CompetitionDistance`: 3 NaN → **median imputation** | Robust to outlier distances |
| `CompetitionOpenSinceMonth/Year`: NaN → median; added **`HasCompetition`** flag | Lets later features ignore fabricated "competition since" dates |
| Promo2 fields: NaN → `0` / `"none"` | Missing ⟺ "never ran Promo2"; binary `Promo2` already encodes membership |
| ~33K absent store-days: investigated → **1 store opened late, 180 have interior closures** | Real retail structure, not corruption; **no data fabricated**; case-study stores must have complete series |
| Sorted by `(Store, Date)`; saved clean snapshots to `data/processed/` | Fast reload for SQL (P7) and dashboard (P8) |

### EDA findings (8 questions → 8 charts)

1. **Growth (headline insight):** network total **fell −5.3%** 2013→2014, BUT sales per
   *open* store-day **rose every year** (6,814 → 7,025 → 7,088) and same-store change is
   **+2.6%**. So the decline was *capacity* (fewer open days), not performance.
2. **Strong weekly seasonality** (Sat/Sun peaks); mid/end-of-month payday bumps.
3. **Store types:** `a` 53.9%, `d` 30.1%, `c` 13.3%, `b` 2.7% of revenue; weekday *pattern*
   similar across types, only the *level* differs → forecast each type separately.
4. **Top 10 stores** (leader ≈ $19.5M lifetime) vs bottom (~$2.2M) → case-study shortlist.
5. **Promo lift: +38.8%** mean sales on promo vs non-promo days (correlational — noted).
6. **State holidays** have measurably higher sales → *expected* deviations, good anomaly context.
7. Biggest network % swings are **explainable** (holidays/promo) → anomaly detection must flag
   *unexplained* deviation, not these.
8. **Sales ↔ Customers correlation = 0.82**, but **Customers cannot be a forecasting feature**
   (future customers are unknown) — leakage-adjacent trap, taught in the notebook.

## 4b. Phase 3 — completed (feature engineering)

Deliverable: `notebooks/03_feature_engineering.ipynb` (31 cells, executed) +
**8 output CSV files** in `data/processed/`.

### Features created (network-level series)

| Category | Count | Features |
|---|---|---|
| Date/Calendar | 12 | day_of_week, month, week_of_year, year, is_weekend, day_of_month, is_month_start, is_month_end, dow_sin, dow_cos, month_sin, month_cos |
| Lagged sales | 5 | sales_lag_1, sales_lag_2, sales_lag_7, sales_lag_14, sales_lag_28 |
| Rolling statistics | 4 | rolling_mean_7, rolling_mean_28, rolling_std_7, rolling_median_7 |
| Promo context | 8 | promo_share, promo_lag_1, promo_lag_7, promo_momentum_7, promo_streak, promo_rolling_7, promo_x_weekend, holiday_x_promo |
| Holiday/School | 4 | any_holiday, state_holiday, any_school_holiday, days_since_holiday |
| Composite | 2 | n_special_flags, split |
| **Total** | **35** | (including target + capacity + split label) |

### Key decisions

- **Cyclical encoding** for day-of-week and month (sin/cos) preserves circular calendar structure.
- **All lagged/rolling features use `shift()`** to prevent data leakage — verified manually in Part G.
- **Temporal split:** Train 2013-01-01 → 2015-06-30 (912 days), Validation 2015-07-01 → 2015-07-15 (15 days), Test 2015-07-16 → 2015-07-31 (16 days).
- **Customers excluded** from all features (future-unknown, leakage risk).
- **Case-study stores:** top 5 by lifetime sales with complete 942-day series.

### Output files

| File | Rows | Purpose |
|---|---|---|
| `network_features.csv` | 942 | Full network series with all features |
| `network_train.csv` | 912 | Training split |
| `network_validation.csv` | 15 | Validation split |
| `network_test.csv` | 16 | Test split |
| `type_features.csv` | 3,768 | Store-type daily series |
| `case_study_features.csv` | 4,710 | Case-study store series |

### Leakage verification (4 checks passed)

1. Lag values match manually computed historical sales ✓
2. Rolling means match manually computed window averages ✓
3. No temporal overlap between splits ✓
4. Validation/Test lag features only reference training data ✓

## 4c. Phase 4 — completed (baseline forecasting)

Deliverable: `notebooks/04_baseline_forecasting.ipynb` (22 cells, executed).

### Results (validation set, 2015-07-01 → 2015-07-15)

| Rank | Baseline | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|---|
| **1** | **Seasonal naive (lag-7)** | **2,011,471** | **2,491,204** | **25.9** |
| 2 | MA-28 (monthly avg) | 2,046,686 | 2,886,823 | 329 |
| 3 | MA-7 (weekly avg) | 2,167,315 | 3,050,504 | 341 |
| 4 | Naive (yesterday) | 2,660,710 | 4,092,755 | 309 |

### Key findings

- **Seasonal naive wins** — confirms the dominant weekly seasonality (EDA Q2). It captures
  the low-sales Sunday pattern (Jul 5 forecast $262K vs actual $269K) that other baselines miss.
- **High MAPE for naive/MA** (~300%+) is dominated by the near-zero-sales Sunday (2015-07-05,
  $269K actual). This is exactly the MAPE-limitation case discussed in the notebook, and why we
  also track MAE/RMSE.
- **MA-7 > MA-28** on RMSE but both are poor forecasters for daily data — they average out the
  very weekly pattern we want to predict.
- Naive (yesterday) is worst: weekend→weekday transitions cause large systematic errors
  (over-predicting Sunday at $6M vs $269K actual).

### What Phase 5 must beat

The bar is **MAE ≈ $2.01M / MAPE ≈ 26%** on validation. Phase 5 models need to add
promo-awareness, holiday-awareness, trend, and interaction effects — the features built
in Phase 3 but not used by these baselines.

## 5. Current project state

- Phases 1, 2, 3, and 4 done; notebooks executed with outputs saved in-file.
- Phase 5 (final forecasting model: SARIMA vs tree-based ML) is next. Awaiting user confirmation
  to begin. statsmodels still NOT installed — install risk on Python 3.14 noted in §7.
- `test.csv` / `sample_submission.csv` exist in `data/raw/` but are **unused** (Kaggle artifacts;
  the project defines its own chronological train/validation/test split).

## 6. Architecture & important files

```
sales-demand-forecasting/            (project root: E:\Data Science\Sales Demand Forecasting and Anomaly Detection)
├── data/
│   ├── raw/                         train.csv (38 MB), test.csv, store.csv, sample_submission.csv  [git-ignored]
│   └── processed/  train_clean.csv (36 MB), store_clean.csv, network_features.csv,
│                   network_train/validation/test.csv, type_features.csv, case_study_features.csv   [git-ignored]
├── notebooks/
│   ├── 01_data_understanding.ipynb  ✅ executed  (~24 cells)
│   ├── 02_eda.ipynb                 ✅ executed  (32 cells, 8 figures)
│   ├── 03_feature_engineering.ipynb ✅ executed  (31 cells)
│   └── 04_baseline_forecasting.ipynb ✅ executed  (22 cells)
├── requirements.txt
├── .gitignore
└── (planned later) notebooks/05..09, src/, app/dashboard.py, sql/analysis.sql, README.md
```

**Key convention:** analysis lives in notebooks; **`src/` consolidation is deferred to
Phase 9** deliberately (avoids premature abstraction); cleaned CSVs carry state between phases.

## 7. Dependencies / setup

- OS: Windows 11 Home; shell PowerShell 5.1; **git NOT installed** (do not install — user decision).
- **Python 3.14.4** (bleeding edge), pip 26.0.1.
- Installed: pandas 3.0.3, numpy 2.5.0, matplotlib 3.11.0, seaborn 0.13.2,
  scikit-learn 1.9.0, scipy, streamlit (present), Jupyter 9.15 / notebook 7.6.
- **statsmodels NOT installed** — Phase 4/5 will `pip install statsmodels` for SARIMA; if no
  wheel for Python 3.14, the **tree-based ML path is the fallback** (final model is chosen on
  validation performance anyway, so this is acceptable).
- `requirements.txt` lists the stack (loose `>=` pins). No venv currently in use.

## 8. Important implementation decisions (do not silently reverse)

1. **Forecast scope:** network-level + **store-type level** (4 series) + a handful of **top
   complete-series stores as case studies**. **No per-store models for 1,115 stores.**
2. **Evaluation:** MAE / RMSE / MAPE on a **chronological train/validation/test split**
   (no random split, no R²).
3. **Models:** naive + moving-average **baselines first** (P4), then pick final between
   **SARIMA and tree-based ML** based on actual validation performance (P5).
4. **Zero/small sales:** closed days (Open=0) are "no sales opportunity", not zero demand;
   `Open` is a model input, not noise. Case-study stores need complete series.
5. **`Customers` is not a forecasting feature** (unknowable in advance). It is historical
   insight only.
6. **Anomaly detection:** statistical baseline (rolling mean/std + Z-score) first; compare
   with Isolation Forest; anomalies = "differs from expected pattern", NOT errors/fraud.
7. **Notebook-first workflow**; relative paths `../data/raw` + a guard `assert` so notebooks
   must be run from the `notebooks/` folder.

## 9. Assumptions made

- DayOfWeek column is reliable (verified: 0 mismatches vs calendar).
- Missing store metadata ⟺ legitimate absence (no competition / never ran Promo2), not corruption.
- Absent store-days are real retail structure (late opens / interior closures); no imputation of sales.
- Promo lift is correlational evidence, not causal.
- Kaggle's `test.csv` is competition scaffolding, not the evaluation set for this project.

## 10. Known bugs / limitations / unfinished items

- **Unfinished:** Phases 5–9.
- **statsmodels install risk** on Python 3.14 (see §7).
- **Env quirk:** `Set-Location` inside a PowerShell tool call persists and moves the harness's
  primary working directory afterward — the next session should **use absolute paths** and not
  rely on `cd`.
- **PowerShell 5.1 gotchas** that bit this session: no `&&`/`||`, no ternary; here-strings
  `@'...'@` must close at column 0; and when building notebook code cells programmatically
  (via Python builder scripts), **`\n` inside a normal Python string becomes a real newline** in
  the generated cell → put any `print("\n...")` in generated source as `print()` + separate line
  to avoid `SyntaxError: unterminated string literal`.
- **Cosmetic:** `zmq` Proactor-loop and kernel-over-TCP warnings during `nbconvert --execute`
  are harmless on Windows.
- git: not installed, no repository, no commits. Nothing to compare against.

## 11. Tests / evaluations performed

- All four notebooks **executed end-to-end** via `jupyter nbconvert --execute` with 0 errors.
- Phase-1/2 numeric verify points: gaps reconciled (33,121 vs 33,120 explained by the single
  late-opened store), DayOfWeek 0 mismatches, promo lift +38.8%, Sales–Customers corr 0.82,
  YoY −5.3% with same-store productivity +2.6%.
- Phase-3 leakage verification: lag values match manual computation, rolling means match manual
  window averages, no temporal overlap between splits, validation/test lag features only
  reference training data. All 4 checks passed.
- Phase-4 baseline results: seasonal naive wins on validation (MAE $2.01M, MAPE 25.9%); MAPE
  ~300% for naive/MA is dominated by the near-zero-sales Sunday on 2015-07-05 (documented).
- `requirements.txt` and `.gitignore` exist; processed CSVs verified by reload (shape match, 0 nulls).

## 12. What NOT to change (already completed & verified)

- `01_data_understanding.ipynb`, `02_eda.ipynb`, `03_feature_engineering.ipynb`, and
  `04_baseline_forecasting.ipynb` — all executed; do not regenerate with different logic unless
  the new phase genuinely requires it.
- The cleaning rules and EDA conclusions in §4 (summaries are embedded in notebook 2's markdown).
- The feature engineering decisions and leakage checks in §4b (summaries are embedded in notebook 3's markdown).
- The dataset choice (Favorita) and the reframed analysis unit (stores, not products).
- The overall scope decisions in §8.
- `data/raw` and `data/processed` contents (regenerate only if raw data changes).

## 13. Exact tasks for Phase 5 — Final forecasting model

Build `notebooks/05_final_forecast_model.ipynb` (next session, after user confirmation):

1. **Try `pip install statsmodels` first.** If a Python 3.14 wheel exists, build a **SARIMA**
   model on the network daily sales (weekly seasonality m=7, likely SARIMA order found via
   ACF/PACF or auto_arima if available). Use validation for tuning, report test-set metrics.
   If install fails, document it and proceed with the ML path only.
2. **Tree-based ML path:** train a **Random Forest / Gradient Boosting (or HistGradientBoosting)**
   regression on the Phase-3 feature table (`network_train.csv` features → `sales` target).
   **Critical discipline:** when predicting the validation/test rows, lagged/rolling features
   must be recomputed from actuals only up to each prediction date (no future actuals); the
   simplest honest approach is to train on features whose lags are all within train and predict
   one day at a time, updating lag features from actuals as you go (iterated/recursive forecast).
3. **Compare against the Phase-4 bar:** seasonal-naive MAE ≈ $2.01M / MAPE ≈ 26% on validation.
   Report MAE/RMSE/MAPE on validation AND test for the chosen model.
4. **Interpretability:** for the ML model, show feature importances (pick the top features);
   for SARIMA, report the fitted parameters and note the seasonal terms.
5. **Model choice write-up:** explain which model won on validation and why; state the final
   test-set numbers; discuss limitations (SARIMA struggles with exogenous promo/holiday effects;
   tree-based models are not truly autoregressive but handle features well).
6. **Save:** persist the chosen model to `data/processed/` (e.g., `final_model.pkl` or joblib)
   so Phase 6 (anomaly) and Phase 8 (dashboard) can reuse it.

Success criteria for P5: at least one real model beats the seasonal-naive baseline on validation;
final numbers reported on the untouched test set; model saved for later phases.

## 14. Context the next session needs

- **User is a Data Science fresher** building this for their resume; wants code that is
  readable/simple/well-commented and full explanations (what problem, why, alternatives,
  limitations, likely interview questions) at each step.
- After each phase, **explain + show how to test, then wait for the user's confirmation**
  before starting the next phase.
- Run everything with Python at the project root; notebooks assert their own folder.
- Produce deliverables at the project root only (write notebooks to `notebooks/`, no temp files
  left behind; the previous session's builder scripts were written to the OS temp dir and removed).
- Current task list in this session mirrors the 9 phases; Phase 5 is `pending`.
- **statsmodels NOT installed** — Phase 5 will `pip install statsmodels` for SARIMA; if no
  wheel for Python 3.14, the **tree-based ML path is the fallback**. The Phase-5 task list §13
  details the exact honest evaluation protocol (recursive forecasting with recomputed lags).