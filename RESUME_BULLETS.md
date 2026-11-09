# Resume bullets — Sales Demand Forecasting & Anomaly Detection

Honest, quantified bullet points. Every number is reproducible from the repo
(`tests/verify_project.py` re-derives the key ones). Pick the ones that fit the role —
**starters** are the simplest to defend in an interview, **deeper** ones show more rigor.

---

## Project one-liner

> Built an end-to-end **sales demand forecasting & anomaly detection system** on the
> Corporación Favorita Kaggle dataset (1M+ store-day records): cleaned the data, engineered
> 35 leakage-free features, compared SARIMA vs gradient boosting, and shipped results as an
> interactive Streamlit dashboard with SQL-based business insights.

---

## Bullets (choose 3–5)

### Starters
- Forecast daily network-level sales from 1,017,209 store-day records (1,115 stores,
  2013–2015) and **cut out-of-sample forecasting error ~84% vs a seasonal-naive baseline**
  (validation MAE $2.01M → **$0.335M**; test MAE **$0.283M**).
- Engineered **35 time-series features** (lagged sales, rolling means, promo/calendar
  interactions) with an explicit **no-leakage discipline**: every feature uses only
  information known the day before, verified by automated checks.
- Built a **5-page Streamlit dashboard** (exec KPIs, store-type analysis, forecasts,
  anomaly alerts, SQL insights) that non-technical stakeholders can use directly.
- Detected **121 anomalous sales days (Z>2)** against model expectations using rolling
  mean/std + Z-score, cross-checked with Isolation Forest.
- Analyzed business drivers in **SQL (SQLite)**: store-type revenue mix, **+39% promo
  lift**, top/bottom store performance, and monthly trends.

### Deeper
- **Compared forecasting families head-to-head**: SARIMA (statsmodels) vs Random Forest vs
  HistGradientBoosting, pick the winner by *actual validation performance*, not fashion —
  HGB won because it can consume promo/holiday features SARIMA ignores.
- **Temporal split**: train 2013–2015-06, validation + test = the last 31 days (no random
  split), so the model is scored exactly the way it will be used — forecasting the future.
- **Honest multi-window evaluation**: the final model is retrained on train+validation;
  the dashboard re-fits a train-only model to score the validation window so every metric
  is out-of-sample for the model that produced it.
- **Anomaly = deviation from expectation, not an error**: framed detection as residual
  Z-scores on the forecasting model, and cross-validated the statistical method against
  Isolation Forest.
- **SQL meets ML**: the same facts are queried live from a SQLite database
  (`sql/analysis.sql`) and surfaced in the dashboard, showing data is not only CSVs.

---

## Stack

Python · pandas · numpy · scikit-learn (HistGradientBoosting, Isolation Forest) ·
statsmodels (SARIMA) · SQLite/SQL · Streamlit · Plotly · Jupyter · matplotlib/seaborn

---

## Anti-bragging guardrail

Everything on this page is either in the repo (number is reproducible) or in its docs
(claim is qualified). Keep the honest caveats handy in interviews:
- The holdout is only 15–16 days → error bars are wide; a rolling-origin backtest would
  tighten them.
- Promo lift is *correlational*, not causal.
- "Anomaly" means "differs from the expected pattern" — not fraud detection.
- The model is store-type/network-level, not product-level.