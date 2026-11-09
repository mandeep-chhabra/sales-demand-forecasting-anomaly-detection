# Interview Preparation Guide — Sales Demand Forecasting & Anomaly Detection

A complete, easy-to-understand guide to this project. Read it top to bottom once, then
drill into the question sections. Everything is written in simple language and every
answer is based on what is actually in this repository — nothing is made up.

> **The one thing to memorise (the "golden thread"):**
> *This project forecasts **one day ahead** using **only information known the day
> before**, and every model is scored on **future data it never saw**. That discipline is
> what makes every number honest.* If you forget everything else, remember this — and
> steer every interview answer back to it.

---

## Part 1 — The Project in Simple Language

### 1.1 The 30-second elevator pitch (memorise this)

> "This is an end-to-end Data Science project on a real retail dataset from Kaggle
> (Corporación Favorita store sales — over 1 million store-day records from 1,115 stores
> over 2.5 years).
>
> I did all the steps of a real project myself: cleaned the data, explored it, engineered
> ~35 time-series features with a strict no-leakage rule, compared three forecasting
> approaches (a simple baseline, SARIMA, and machine-learning models), and picked the
> winner by actual performance on a held-out future window. The final model reduced
> forecasting error by about **84%** compared to a strong baseline.
>
> Then I built an **anomaly detection system** that flags days where sales behave
> unusually compared to what the model expected, analysed the business with **SQL**, and
> shipped everything in an interactive **Streamlit dashboard** that non-technical people
> can use.
>
> The whole thing is reproducible — running one test script passes 10 automated checks."

### 1.2 What problem are we solving? (the business story)

A retail company (Corporación Favorita, Ecuador) wants answers to two questions:

1. **Forecasting:** What will today's / this week's sales be, so they can plan stock,
   staff, and promotions? We forecast at the whole-network (all stores combined) level,
   and also by store type and for a few top stores.
2. **Anomaly detection:** When something unusual happens in sales — a sudden spike or a
   sudden drop — it's worth knowing quickly. A spike might mean a payday or a promotion
   worked; a drop might mean a data error or a store problem. We flag those days for a
   human to look at.

The deep point: **forecasting and anomaly detection are the same idea from two directions.**
A forecast tells you the *expected* pattern. An anomaly is a day that *doesn't fit* the
expected pattern. That's why we use the forecast model to detect anomalies.

### 1.3 The data

| Fact | Value |
|---|---|
| Source | Kaggle competition "Store Sales — Time Series Forecasting" (Corporación Favorita) |
| Rows | 1,017,209 store-day records |
| Stores | 1,115 stores |
| Time range | 1 Jan 2013 → 31 Jul 2015 (942 days) |
| Level of analysis | store / store-type / promo-period — **not** product level |
| Key columns | Date, Store, Sales (target), Customers, Promo, StateHoliday, StoreType, Open, etc. |

### 1.4 What I built — the 8 phases at a glance

| # | Phase | What it did | Key output |
|---|---|---|---|
| 1 | Data understanding | Looked at the raw files, figured out their meaning | `notebooks/01` |
| 2 | Cleaning & EDA | Fixed bugs, explored patterns, found the story | `notebooks/02` + clean CSVs |
| 3 | Feature engineering | Built ~35 time-series features, leakage-checked | `notebooks/03` |
| 4 | Baselines | Tested simple forecasting rules first | `notebooks/04` |
| 5 | Final forecast model | SARIMA vs ML → HistGradientBoosting wins | `notebooks/05` |
| 6 | Anomaly detection | Z-score on residuals + Isolation Forest cross-check | `notebooks/06` |
| 7 | SQL analysis | 30+ business queries on a real SQLite database | `notebooks/07`, `sql/analysis.sql` |
| 8 | Dashboard | Interactive 5-page Streamlit app | `app/dashboard.py` |

### 1.5 Key results (the numbers that matter)

1. The ML model (HistGradientBoosting) reached **validation MAE $0.335M** and **test MAE
   $0.283M**, vs **$2.01M** for the best baseline → **~84% error reduction**.
2. SARIMA (a classic time-series model) reached $0.649M — good, but it can't see
   promotions/holidays, which is why ML won.
3. **121 of 942 days** (about 13%) were flagged as anomalies at |Z|>2 — deviations worth a
   human look, not errors.
4. Promotions lift sales on average ~39% (correlational, not causal).
5. Revenue fell −5.3% year-over-year, but that was a *capacity* story (shorter/closed
   periods), not performance — sales per open store-day actually rose +2.6%.
6. Sales and Customers are strongly correlated (0.82), but Customers is **not** used as a
   forecasting feature because it's unknown at prediction time (leakage).

### 1.6 The "why this project is good" summary

- **End-to-end:** CSV → cleaned data → features → model → anomaly detection → SQL →
  dashboard → automated tests. Nothing is left as an unsolved loose end.
- **Honest methodology:** chronological split (not random), future data never seen by any
  model, and a tricky evaluation subtlety handled correctly (see §3.6, the "interview trick").
- **Good decisions over fancy stack:** no Docker/Kafka/Spark — the value is in *correct
  thinking*, which is exactly what interviewers test.

---

## Part 2 — Simple Explanations of Every Concept

> These are short plain-English explanations. If an interviewer asks any concept below,
> this is how you'd explain it conversationally. Deepening each one is the job of Part 3.

### 2.1 What is a time series?

A time series is just data measured over time in order — e.g., the total sales on each day
from 2013 to 2015. Forecasting = predicting the future values of this ordered sequence.

Three common patterns in sales time series:
- **Trend:** long-term up or down direction.
- **Seasonality:** a repeating pattern — here, a strong *weekly* cycle (weekends sell more
  than weekdays, Sundays nearly close).
- **Noise/irregularity:** random day-to-day ups and downs.

### 2.2 Data leakage (the single most important concept)

**Leakage** = a model accidentally seeing information it would not actually have at the
time of prediction. Example: if I use "today's customers" to predict "today's sales",
the model looks great on training data but is useless in production, because you don't
know how many customers came until the day is already over.

Ways this project avoids leakage:
1. **Lag features** use `shift()` — they only reference *past* days.
2. **Rolling statistics** are computed with `shift()` before rolling, so row *t* only sees
   days *t-1, t-2, ...*.
3. **Chronological split** — always train on the past, test on the future. Never shuffle.
4. **Customers is excluded** as a feature.
5. **4 automated checks** in the feature notebook verify no future information slipped in.

### 2.3 The one-day-ahead, only-past-information discipline

We forecast **one day at a time**. For each day we predict, all features are computed only
from data known *before that day*. This is how the model will actually be used in real
life (you predict tomorrow using everything up to today). This single idea appears in
every phase.

### 2.4 Feature engineering (what the model learns from)

A "feature" is a column of information the model can use. We engineered ~35 features:

| Group | Examples | Why |
|---|---|---|
| Date/calendar | day_of_week, is_weekend, month, year | Sales follow the calendar |
| Lags | sales_lag_7 (sales this day last week), sales_lag_1 | The best predictor of today is the recent past |
| Rolling stats | rolling_mean_7, rolling_std_7 | Recent level and variability |
| Promo context | promo_share (fraction of stores running a promo) | Promotions drive sales |
| Holiday | state_holiday, one-hot flags | Holidays change behaviour |
| Interactions | promo × weekend | Two effects combined beat each alone |
| Cyclical encoding | sin/cos of day_of_week and month | See §2.5 |

### 2.5 Why cyclic (sin/cos) encoding for weekday/month?

Days of the week are a circle: Sunday (7) is *next to* Monday (1), but as plain integers
7 and 1 are far apart. If the model sees numbers, it thinks Sunday and Monday are very
different days. Sine/cosine encoding maps the day onto a circle, so the model understands
"distance around the week" instead of "the number is bigger/smaller." Same trick for
months (December is next to January).

### 2.6 Baselines (why bother?)

Before building a fancy model, ask: *what's the dumbest reasonable forecast?* We tested:
- **Yesterday-naive:** "today = yesterday". Bad because Monday vs Sunday are totally different.
- **Moving average:** "today = average of last N days". Smooths away the weekly pattern.
- **Seasonal naive:** "today = same day last week" (lag-7). The winner, because the series
  is strongly weekly — validation MAE $2.01M.

If a fancy model can't beat seasonal-naive, it's worthless. The gap between $2.01M and
$0.335M is exactly the value the features + model added.

### 2.7 The models

- **SARIMA** — the classic statistics time-series model. It learns the series from its own
  past values and past errors, plus explicit weekly seasonality. Notation
  `SARIMA(p,d,q)(P,D,Q,m)` = non-seasonal and seasonal orders. Ours:
  `SARIMA(1,0,1)(1,1,1,7)` — weekly (m=7) seasonal differencing. **Limitation:** it
  ignores external inputs (promotions, holidays), which is why ML beat it.
- **Random Forest** — averages many decision trees. Robust but a bit behind HGB here.
- **HistGradientBoosting (HGB)** — a modern, fast gradient-boosted trees model. Usually
  the best tabular predictor. **Won** because it *can* see all the promo/calendar features
  SARIMA can't.

Important framing: these ML models are not "time-series models" by themselves — they are
regressors. They become forecasters because we (a) give them lag features and (b) evaluate
them on future windows.

### 2.8 Evaluation — how do we know it works?

- **Chronological split:** train = 2013-01-01 → 2015-06-30; validation = 2015-07-01 → 15;
  test = 2015-07-16 → 31. The last ~31 days are never used for training decisions about
  the final numbers except the final test report.
- **MAE** (Mean Absolute Error): average |actual − predicted|. In dollars. This is the
  main metric.
- **RMSE** (Root Mean Squared Error): like MAE but punishes big errors harder.
- **MAPE** (Mean Absolute Percentage Error): percent error. Reported with a caveat — see
  §3.7.
- We deliberately **do not** use R² because for forecasting what matters is absolute error
  on the future, and because the data is not a random sample — you can't cross-validate it
  randomly.

### 2.9 Anomaly detection (simple version)

1. Fit the forecast model → it produces an *expected* value for each day.
2. residual = actual − expected (how far reality was from the prediction).
3. Compute a **rolling Z-score**: how many "standard deviations" the residual is from the
   recent normal range of residuals (window = 7 days, using only past data via shift(1)).
4. Flag the day if |Z| > 2 → **121 days** flagged.
5. **Cross-check** with Isolation Forest (an unsupervised tree-based anomaly detector) —
   the two methods broadly agree, which increases confidence.

Why score residuals, not raw sales? Raw sales spike every weekend and every promo — a
huge spike is *expected*, not anomalous. By scoring the *residual* (after removing the
expected pattern), we flag only days that deviate *beyond* what the features explain.

### 2.10 SQL & the dashboard

- **SQL:** The cleaned data lives in a real SQLite database. `sql/analysis.sql` has 30+
  business queries — revenue by store type, promo lift, top/bottom stores, monthly trends,
  window functions like `NTILE`, `CASE` bucketing. This shows dataset skills beyond pandas.
- **Dashboard:** A Streamlit app with 5 pages — Overview (KPIs, seasonality), Store Types,
  Forecasting (honest out-of-sample results + feature importance), Anomalies, and SQL
  Insights. Each chart has a caption explaining *why it matters*, not just what it shows.

---

## Part 3 — Interview Questions and Answers

> Each answer is written the way you'd say it out loud: short, concrete, with the "why".
> The 💡 markers highlight especially-likely questions.

### 3A. Opening questions (likely the FIRST thing asked)

**Q: Tell me about this project. What did you build and why?**
See §1.1. Add one sentence on motivation: "I wanted a real end-to-end project that shows I
can take raw data to a usable product, not just a model in a notebook."

**Q: What was your role? / How much of this did you do yourself?**
All of it, from data download to dashboard to tests. I made the methodology decisions and
documented the trade-offs in README.md / INTERVIEW_PREP.md.

**Q: Why did you pick this dataset?**
It's a real Kaggle retail competition with a rich business story — promotions, holidays,
store types, a strong weekly seasonality, and over a million rows. It exercises nearly
every core Data Science skill without needing big-data infrastructure.

**Q: In one sentence, what did you actually achieve?**
A forecast system that predicts tomorrow's network sales ~84% more accurately than a strong
baseline, plus an anomaly alert system, delivered as an interactive dashboard.

**Q: What's the single most important thing an interviewer should know about your project?**
The one-day-ahead, only-past-information, honest-holdout discipline. Everything else —
models, features, anomaly flags — follows from that.

---

### 3B. Data understanding & cleaning (Phases 1–2)

**💡 Q: What does "cleaning the data" actually mean for time series?**
Three distinct things:
1. **Fix representation bugs** — e.g., `StateHoliday` was mixed-type: 855k rows were the
   string `'0'` and 131k were the integer `0`; pandas threw `DtypeWarning`. Fixed by
   coercing to string and mapping `'0' → 'none'` while keeping `a/b`/`c` for public
   holidays.
2. **Decide what missing means** — e.g., missing promo-2 metadata means "no promo", not
   corruption. ~33k "missing" store-days were real retail structure (a store that opened
   late, interior closures), not data problems.
3. **Never impute the target.** Closed-day rows (Open=0, Sales=0) are "no sales
   opportunity", not "zero demand". We keep them and let the model learn `Open` as a
   feature.

**Q: Why not fill every missing store-day with 0 sales?**
Because absence isn't zero *demand*. Filling them would teach the model big cliffs of zeros
where a store happened to be closed, and would bias every aggregate. The honest fix is to
respect the calendar structure and use `Open` as a model input.

**💡 Q: Data quality — what issues did you actually find?**
- Mixed-type `StateHoliday` (string `'0'` vs int `0`) → `DtypeWarning`.
- ~33k store-days absent from the raw file → explained by store opening dates and interior
  closures, verified against the data itself.
- Decided `Promo2` missing ⟺ "never ran".
- Confirmed `Sales` has no nulls in its domain (Open stores), so no imputation needed.

**Q: How many rows/stores/days and what's the time span?**
1,017,209 rows · 1,115 stores · 942 days · 2013-01-01 → 2015-07-31.

---

### 3C. EDA & business insights (Phase 2)

**💡 Q: What three insights drove the rest of the project?**
1. **Weekly seasonality is the dominant pattern** (weekends peak, Sundays near-zero).
   → So the baseline is lag-7 (same day last week).
2. **Store type drives the level, not the shape** of sales.
   → Forecast by store type, not 1,115 separate stores.
3. **Promotions lift sales ~39%; holidays change behaviour.**
   → These are *expected* deviations, so anomaly detection must flag anything *beyond them*.

**💡 Q: Sales fell −5.3% year-over-year. Is that bad news?**
Not as it sounds. Same-store sales per open day rose **+2.6%**, and productivity per
open-store-day rose every year (6,814 → 7,025 → 7,088). The revenue drop was *capacity* —
fewer store-days open (a store closed for interior work, a later-opening store) — not
underperformance. The lesson: **measure the right unit.** Revenue fell but the business got
more productive.

**Q: Why can't `Customers` be a forecasting feature?**
Because it's only known *after* the day happens. Its correlation with sales is 0.82 — which
makes it tempting — but using it would leak the future. We use it as historical insight only.

**Q: How did you handle the closed-day vs zero-sales question in EDA?**
We treat Open=0 as "no opportunity" and study patterns on open days, but keep all rows for
modeling so the model learns when stores are closed.

---

### 3D. Feature engineering & leakage (Phase 3)

**💡 Q: What is data leakage and how did you avoid it?**
Leakage = the model seeing information it wouldn't have at prediction time. I avoided it
with (a) `shift()`-only lag/rolling features, (b) 4 automated checks confirming lag values
equal the manually-computed past actuals and that validation/test never reference future
rows, (c) a chronological split, and (d) excluding `Customers`.

**Q: Walk me through your features.**
~35 features in groups: date/calendar (day_of_week, is_weekend, month, sinusoidal
encodings), lags (sales_lag_1, sales_lag_7), rolling stats (rolling_mean_7, rolling_std_7,
computed with shift so nothing leaks), promo context (promo_share = fraction of stores on
promo), holiday flags (one-hot for state holiday classes), and interactions
(promo × weekend).

**💡 Q: Why circular sin/cos encoding for day-of-week and month?**
Sunday and Monday are adjacent on the calendar but 7 and 1 are far apart as integers, so a
linear encoding is wrong. Sin/cos puts days on a circle so the model sees "distance around
the week", not "a bigger number". Same for months (Dec → Jan wraps).

**Q: Won't ~35 features overfit on ~900 rows?**
The model is regularised gradient boosting, and — more importantly — we only judge it on
held-out future windows, not training accuracy. We report validation and test MAE, not R²
on training.

**Q: How did you choose the train/validation/test split dates?**
Chronologically: train through 2015-06-30, then the last ~31 days split as 15 validation
+ 16 test. That leaves 4+ weekly cycles in training and keeps enough holdout days to see a
full week or two of patterns. No random split — random would leak because adjacent days are
dependent.

**Q: What is a lag feature and why is it powerful?**
`sales_lag_7` = the sales value from 7 days ago (one week earlier). It's the single most
important feature because weekly patterns repeat — today's sales resemble this day last week.
Lagging with `shift(7)` means at prediction time the value is already known — no leakage.

---

### 3E. Baselines (Phase 4)

**💡 Q: Why build baselines at all?**
A model is only worth what it adds over a cheap heuristic. The bar was seasonal-naive
(lag-7): MAE $2.01M. If a real model can't beat that, it's not worth the complexity.

**Q: Why did seasonal-naive beat yesterday-naive and moving averages?**
The series is strongly *weekly*. "Today ≈ same day last week" captures the weekend/Sunday
structure. Yesterday-naive overshoots Sundays (predicting Saturday's high number), and
moving averages smooth away the very weekly pattern we want to predict.

**Q: MAPE blew up to ~300% for the naive baselines. Is that a problem?**
It's a known MAPE limitation: near-zero Sunday sales turn a tiny absolute error into a huge
percentage error. That's exactly why we report MAE and RMSE too — MAE is in dollars and is
dominated by the absolute error, which is what a business actually cares about. (Saying
this out loud scores points — it shows you understand metric pitfalls.)

**Q: What exactly did the baseline set up for the rest of the project?**
A number to beat ($2.01M) and a comparison frame: the final model's $0.335M shows the real
value the features and model added.

---

### 3F. The forecasting model (Phase 5) — THE main section

**💡 Q: Why both SARIMA and machine learning? Why not just one?**
They encode time differently and have opposite strengths:
- SARIMA is a true time-series model with explicit weekly seasonality, but it ignores
  promotions/holidays (no exogenous inputs).
- Tree-based ML uses all 32–35 features (promo, holidays, interactions) but isn't a
  time-series model on its own — you feed it lags and evaluate it on the future.
I let the data decide on validation: **HGB $0.335M vs SARIMA $0.649M vs seasonal-naive
$2.01M.**

**Q: What exactly did the ML model learn over the baseline?**
It combines lags + rolling means + promo share + weekend + holiday interactions — e.g., it
learned that *a promo on a weekend lifts sales more than either alone*. The baseline only
knows "same day last week". That combinatory power is where the ~84% error reduction comes
from.

**Q: Why is a Random Forest / Gradient Boosting a "forecasting" model at all? Fair challenge.**
It's *not* autoregressive by construction — it's a tabular regressor. It becomes a
forecaster because we (1) feed it lagged/rolling features that encode time, and (2) evaluate
it strictly on future windows with a one-day-ahead setup. The honest framing is
"feature-based, one-day-ahead, evaluated out-of-sample."

**Q: How did you pick the SARIMA order?**
`SARIMA(1,0,1)(1,1,1,7)` — seasonal differencing with period m=7 for the weekly cycle,
order choices checked via ACF/PACF and validated on the validation window. No `auto_arima`
needed.

**💡 Q: HistGradientBoosting has no `feature_importances_`. How did you get feature importance?**
Correct — HGB doesn't expose native importances, so I used **permutation importance**
(shuffle each feature's values and measure how much error increases; a big increase = an
important feature). Top features: `sales_lag_7`, `rolling_mean_7`, `promo_share`,
`is_weekend`, `sales_lag_1` — the weekly rhythm plus promo, which exactly matches the EDA
story. Knowing your tool's limits is a good signal.

**💡 Q: You report validation MAE $0.335M and test MAE $0.283M. What's the subtlety?**
This is the trickiest methodological point in the project — get it right and you stand out:
- If you retrain the winner on **train + validation** before scoring, that model is
  *in-sample on validation* (it was trained on those very days) — its validation MAE is
  misleadingly great (~$0.026M).
- So: the **validation** number ($0.335M) comes from a **train-only** model.
- The **test** number ($0.283M) comes from the final model retrained on **train+validation**
  — it never saw test.
The dashboard enforces this: it caches a train-only refit to score validation, and uses the
saved model for test. This is the "honest evaluation" trick — it separates a learned-out
number from a defensible one. Practise saying it: *"every number I report is out-of-sample
for the model that produced it."*

**Q: What were the hyperparameters?**
Random Forest: n_estimators=500. HistGradientBoosting: max_iter=300. Both random_state=42.
Models were chosen on *validation* performance — the test set never influenced a decision.

**Q: Did you tune hyperparameters?**
Not exhaustively — the stock estimators already beat the baseline by a wide margin, and
with only ~880 training rows heavy tuning would risk overfitting the validation window.
That's an honest answer; we optimised for methodology clarity, not squeezing 2% more.

**Q: Why MAE/RMSE/MAPE and not R²?**
R² is about variance explained on the data you have; forecasting is about absolute error on
the future. Also, a random train/test split (which R² cross-validation assumes) is invalid
for time series. MAE is in dollars, so it translates directly to business cost; RMSE shows
whether big misses are punished; MAPE is percentage error but distorts near-zero days.

---

### 3G. Other model/eval questions you might get

**Q: What if a store runs no promo today vs everyone runs one — explain promo_share.**
`promo_share` is the fraction of open stores that ran a promotion that day. At network
level it tells the model "how promotional is today" — a strong driver, since promos lift
sales ~39% on average and the effect is stronger on weekends.

**Q: Why not build a model for each of the 1,115 stores?**
Too many tiny noisy series to model well, and it's not what the business question asked
(network/type level is the planning level). We model the network, the 4 store types, and a
handful of top stores as case studies. Grouping by type captures the "level" difference
without 1,115 models.

**Q: How would you forecast one day further — beyond one-day-ahead?**
Honest answer: this project is one-day-ahead. For multi-step, I'd switch to either
recursive prediction (feed predicted values back as lags), direct multi-step training, or a
rolling-origin backtest. That's on the "next steps" list.

**Q: What does the dashboard's forecasting page show?**
Actual vs predicted for the out-of-sample windows, the error metrics per window (labelled
with which model scored which window and why — so the numbers are honest), feature
importance, and the test-window forecast chart.

---

### 3H. Anomaly detection (Phase 6)

**💡 Q: What does "anomaly" mean in this project?**
A day whose sales differ from what the forecast model expected — the residual is large
relative to recent residual variation. It is **not** fraud detection and **not** an error
flag; it's a triage signal for analysts (payday spike, promo failure, data glitch).

**💡 Q: Why score Z-scores on residuals, not on raw sales?**
Raw sales spike every weekend and every promo — a spike is *expected*, not anomalous. By
scoring the residual (actual − expected), weekends and promos are already subtracted out,
so the Z-score flags deviations *beyond* the expected pattern. This is the key framing.

**Q: Walk me through the Z-score method, exactly.**
1. Predict every day with the model (retrained on train+valid). residual = actual − pred.
2. For each day, take rolling mean and std of residuals using a window of 7 days, with
   `shift(1)` so day *t* only uses residuals from *before* day *t*.
3. z = (residual_t − roll_mean) / roll_std.
4. Flag if |z| > 2 → 121 days flagged (about 13%); also count |z| > 3.
5. The last 16-day test window had no extreme anomalies — the model is well-calibrated.

**Q: Why also use Isolation Forest?**
As an independent sanity check. It's unsupervised and tree-based (isolates rare points in
feature space), no threshold tuning, but less interpretable. It ran with contamination=0.02
on features like residual, rolling stats, day_of_week, promo_share, is_weekend. The two
methods broadly agreed, which increases confidence. The Z-score stays the primary because
business users need a sentence like "today's sales were 3.2 standard deviations below
expectation."

**Q: What did the flagged anomalies look like — any patterns?**
They cluster around known causes — paydays, holidays, promo swings — i.e., real-world events
the model's features didn't fully explain, not random noise.

**Q: Why window = 7 for the rolling Z-score?**
It matches the weekly cycle: 7 days of residuals give a stable "normal recent behaviour"
estimate while still reacting to shifts. Too long → slow to adapt; too short → noisy.

**Q: What would you improve in anomaly detection?**
Add seasonality-aware thresholds, use the model's own residual distribution more formally,
and in production alert via the dashboard daily.

---

### 3I. SQL & the dashboard (Phases 7–8)

**Q: Why is SQL in a Data Science project?**
Real pipelines read from databases, not just CSVs. SQLite makes the analysis reproducible,
queryable, and auditable — `sql/analysis.sql` contains 30+ business queries. It also shows
window functions (`NTILE` deciles), `CASE` bucketing (competition-distance bands), and
executive-level aggregation — skills beyond `.groupby()`.

**Q: Give an example of an interesting SQL insight.**
Revenue mix by store type; annual trend per store type; promo lift per type; top/bottom
stores by sales; monthly network trends; stores running promotions most aggressively;
customers-to-sales relationship per type — pulled live from the database and surfaced in
the dashboard's SQL Insights page.

**Q: What makes the dashboard "explainable"?**
Every page explains *why* the number matters — each chart has a caption. The Forecasting
page labels which model scored which window and why. For a non-technical stakeholder, the
"so-what" *is* the product.

**Q: How does the dashboard stay fast with ~1M rows?**
It loads the cleaned, processed artifacts (not raw), loads everything once and caches it,
and queries the pre-aggregated tables. It never re-trains a model.

---

### 3J. Business impact & "so what?" questions

**Q: What is the business value of all of this?**
Cheaper operations: more accurate demand forecasts mean better stock levels, fewer
out-of-stocks, less waste, better staffing plans. The anomaly alerts mean unexpected events
are caught within a day instead of a week. The numbers you can quote: ~84% lower forecast
error; 121 event days flagged for attention; ~39% promo lift understood.

**Q: How would you quantify the dollar impact of a better forecast?**
MAE is the average dollar miss per day. Going from ~$2.01M to ~$0.335M average daily error
is a large, concrete reduction — you'd take the daily miss, multiply by days, and estimate
inventory/staff savings. Framing MAE in dollars is the business translation.

**Q: Is the +39% promo lift causal?**
No — it's correlational. Promo days may coincide with weekends/holidays. To claim causality
I'd need a proper experiment (design, controls, A/B or regression adjustment). I say this
honestly because it shows methodology maturity.

---

### 3K. Limitations & honest future work

**💡 Q: What are the project's limitations? (They WILL ask this.)**
- The holdout is only 15–16 days → error bars are wide. A rolling-origin / backtest would
  tighten them.
- Network/type level only — no per-store or per-product model.
- Promo lift is correlational, not causal.
- "Anomaly" is a triage signal, not proven fraud/error.
- SARIMA is univariate (no exogenous features); SARIMAX could add promos/holidays.
- Mape distortion on near-zero days.

**Q: What would you do differently with more time?**
1. **Rolling-origin / backtesting** for robust error bars.
2. **SARIMAX** to give the statistical model the promo/holiday inputs too.
3. **Store clustering** — one model per similar-store cluster instead of whole-network.
4. **Two-stage model** using a *Customers* forecast as a feature, carefully.
5. **Real holiday calendar** (e.g., the `holidays` package) instead of generic flags.
6. **Multi-step forecasting** (recursive or direct) and richer evaluation (crps, coverage).

**Q: How would you deploy this in production?**
Modest, honest answer: wrap the trained model + feature builder in an API, run scoring on a
schedule (e.g., a daily job that predicts the next day), send anomalies to the dashboard,
and monitor data drift. I deliberately kept this a local-end-to-end project (no Docker/Spark)
because the goal was correct methodology first.

---

### 3L. Behavioral & "tell me about a challenge" questions

**Q: Tell me about a bug or challenge you hit and how you solved it.**
Several good ones to choose from:
- **The `Customers` leakage trap:** almost added it as a feature (0.82 correlation) until I
  realised it's unknowable at prediction time. I removed it and documented why — a
  future-proofing decision.
- **The mixed-type `StateHoliday` bug:** a real pandas warning, diagnosed and normalised.
- **The honest-evaluation subtlety:** the saved model was in-sample on validation (MAE
  $0.026M, misleading). I fixed the dashboard to score validation with a train-only refit so
  every number is honest. This is the one interviewers love.

**Q: How did you keep yourself honest about results?**
Automated checks (10 end-to-end assertions), a chronological split, explicit leakage checks,
documenting limitations in README, and never letting the test set influence a decision.

**Q: How do you decide when a task is "done"?**
When the pipeline runs end-to-end, tests pass, numbers are reproducible, and the story is
explainable to a non-technical person. Documentation is part of done.

**Q: How would you explain this project to a non-technical manager?**
Use the business story, not the models: "We predict sales more accurately to plan stock and
staff, and we get an alert whenever actual sales drift far from expectations. Every chart in
the dashboard answers a business question." Avoid jargon; use the 30-second pitch from §1.1.

---

### 3M. Rapid-fire one-liners (memorise these)

| Question | One-liner |
|---|---|
| Dataset? | Corporación Favorita Kaggle store sales — 1M+ rows, 1,115 stores, 942 days. |
| Best model? | HistGradientBoosting, validation MAE $0.335M, test $0.283M. |
| Baselines? | Seasonal-naive (lag-7) $2.01M; SARIMA $0.649M. |
| Error reduction? | ~84% vs the best baseline. |
| Key features? | sales_lag_7, rolling_mean_7, promo_share, is_weekend, sales_lag_1. |
| Anomalies? | 121/942 days at |Z|>2; Z-score on residuals cross-checked with Isolation Forest. |
| Promo lift? | ~39% average (correlational, not causal). |
| Split? | Chronological: train ≤ 2015-06-30, then 15 validation + 16 test days. |
| Leakage? | Lag/rolling via shift(), Customers excluded, 4 automated checks. |
| Dashboard? | Streamlit, 5 pages, honest out-of-sample metrics. |

---

## Part 4 — Quick Numbers to Have Cold

| Quantity | Value |
|---|---|
| Data | 1,017,209 rows · 1,115 stores · 942 days |
| Baseline (seasonal naive) | Validation MAE $2.01M |
| HistGradientBoosting | Validation MAE $0.335M · Test MAE $0.283M |
| SARIMA(1,0,1)(1,1,1,7) | Validation MAE $0.649M · Test MAE $0.969M |
| Error cut vs baseline | ≈ 84% (val) / 86% (test) |
| Feature count | ~35 engineered (32–35 used by the model depending on encoding) |
| Top features | sales_lag_7, rolling_mean_7, promo_share, is_weekend, sales_lag_1 |
| Promo lift | +38.8% mean (correlational) |
| Anomalies (|Z|>2) | 121 / 942 days (12.8%) |
| Sales–Customers correlation | 0.82 (Customers NOT a feature) |
| Year-over-year | Revenue −5.3%, same-store productivity +2.6% |
| Rolling Z-score window | 7 days with shift(1) (no leakage) |
| Isolation Forest | contamination=0.02, n_estimators=200 |
| Random Forest | n_estimators=500, random_state=42 |
| HGB | max_iter=300, random_state=42 |

---

## Part 5 — Questions YOU Can Ask the Interviewer

Smart follow-ups show interest and confidence. Pick 1–2.

- "How does the team measure forecast accuracy in production, and how often does it review
  model performance?"
- "Do you run demand forecasting at store level or product level, and what's the planning
  horizon?"
- "What does the end-to-end pipeline look like today, and where would a new hire plug in?"
- "How does the team handle a new promotion or a new store — is that a modelling problem
  or a process problem?"
- "What would you look for in my project that I haven't mentioned yet?"

---

## Part 6 — Concepts You Should Also Be Ready For (quick prep)

Interviewers pivot from your project to *theory*. Get these basics right:

- **Mean / variance / skew of the target:** sales are right-skewed, heavy on weekends.
- **Stationarity & differencing:** SARIMA needs stationarity; seasonal differencing (period
  7) addresses the weekly cycle.
- **ACF/PACF:** tools used to choose SARIMA orders (seasonal and non-seasonal).
- **Cross-validation for time series:** normal K-fold is wrong; use time-series splits or
  rolling origin / walk-forward.
- **Bias–variance trade-off:** more features → variance risk; more data → less relying on
  a single holdout.
- **Tree ensembles vs linear/statistical models:** trees handle interactions and scaling
  well but don't extrapolate beyond the training range (a real limitation for
  trend-heavy forecasts).
- **Overfitting signs:** great train metrics, poor validation; feature importance unstable;
  wild metric swings across small holdouts.
- **Metrics:** MAE (robust, dollars), RMSE (punishes big errors), MAPE (share-based but
  unstable at zeros), and why we avoid R² here.
- **Correlation vs causation:** the +39% promo lift must not be claimed as causal.

---

*Remember: interviews on this project are won in the "why" — why this feature, why this
split, why this metric, why this model. Always answer with the reason, and always steer
back to the golden thread: honest, one-day-ahead, only-past-information forecasting.*