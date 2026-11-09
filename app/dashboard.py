"""
Sales Demand Forecasting & Anomaly Detection — Streamlit Dashboard
====================================================================

Interactive dashboard for the end-to-end project. Four pages:

1. **Overview**      — headline KPIs, network sales trend, weekly seasonality
2. **Store types**   — performance by store type, promo impact, top stores
3. **Forecasting**   — final model (HistGradientBoosting) actual vs predicted,
                       feature importances, honest out-of-sample results
4. **Anomalies**     — rolling-mean Z-score anomaly flags over time
5. **SQL Insights**  — business answers straight from the SQLite database

Run from the project root (first time may install streamlit):

    pip install streamlit
    streamlit run app/dashboard.py

The dashboard reads only the clean, processed artifacts produced by the
notebooks (Phases 1-6) — it never re-trains a model. It loads everything once
and caches it, so it stays fast even though the sales table holds ~1M rows.
"""

import os
import joblib
import pandas as pd
import numpy as np
import sqlite3

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------------
# Configuration: resolve data paths relative to this file, so the app works
# no matter which directory the user launches streamlit from.
# ----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))          # .../app
ROOT = os.path.dirname(APP_DIR)                               # project root
PROC = os.path.join(ROOT, "data", "processed")
DB_PATH = os.path.join(PROC, "sales.db")
MODEL_PATH = os.path.join(PROC, "final_model.pkl")

st.set_page_config(page_title="Favorita Sales Dashboard", layout="wide",
                   page_icon="📊")

# ----------------------------------------------------------------------------
# Consistent colors / style helpers
# ----------------------------------------------------------------------------
SALES_COLOR = "#1f77b4"
PRED_COLOR = "#ff7f0e"
ANOM_COLOR = "#d62728"


# ----------------------------------------------------------------------------
# Data loading (cached — runs once per app session)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_network() -> pd.DataFrame:
    """Network-level daily series + all Phase-3 features (942 rows)."""
    df = pd.read_csv(os.path.join(PROC, "network_features.csv"),
                     parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_types() -> pd.DataFrame:
    """Store-type daily series (3,768 rows: 4 types x 942 days)."""
    df = pd.read_csv(os.path.join(PROC, "type_features.csv"),
                     parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_case_studies() -> pd.DataFrame:
    """Top-5 case-study stores (4,710 rows)."""
    df = pd.read_csv(os.path.join(PROC, "case_study_features.csv"),
                     parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    """Network-level anomaly scores (942 rows)."""
    df = pd.read_csv(os.path.join(PROC, "anomaly_scores.csv"),
                     parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_store_clean() -> pd.DataFrame:
    """Store metadata (1,115 rows)."""
    return pd.read_csv(os.path.join(PROC, "store_clean.csv"))


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved HistGradientBoosting model + its feature list."""
    return joblib.load(MODEL_PATH)


# ----------------------------------------------------------------------------
# Feature reconstruction — mirrors notebook 05 EXACTLY:
#   drop (Date, sales, split, n_open, year, state_holiday)
#   one-hot encode state_holiday with prefix "hol"
#   select the 32 columns the model was trained on, in its order.
# ----------------------------------------------------------------------------
DROP_COLS = ["Date", "sales", "split", "n_open", "year", "state_holiday"]


def build_feature_frame(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Return X with the model's exact feature columns, in the right order."""
    base = [c for c in df.columns if c not in DROP_COLS]
    dummies = pd.get_dummies(df["state_holiday"], prefix="hol", dtype=int)
    full = pd.concat([df[base].reset_index(drop=True), dummies.reset_index(drop=True)],
                     axis=1)
    full = full[[c for c in features if c in full.columns]]
    return full


@st.cache_resource(show_spinner=False)
def train_only_forecast():
    """Re-fit the SAME HistGradientBoosting on TRAIN-ONLY data only.

    Why? The saved model was retrained on train + validation (standard "final
    model" practice, see notebook 05). That makes it *in-sample* on the
    validation window, so it cannot honestly score it. To present the validation
    period out-of-sample we re-fit an identical model on train-only and predict
    every day with it — exactly the protocol that produced the reported
    validation MAE of $0.335M. Cached once per app session.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    net = load_network()
    saved = load_model()
    features = saved["features"]
    X = build_feature_frame(net, features)

    tr_mask = net["split"].eq("train").values
    Xtr, ytr = X[tr_mask], net.loc[tr_mask, "sales"]
    keep = ~Xtr.isna().any(axis=1)          # drop warm-up rows with no lag history
    model = HistGradientBoostingRegressor(random_state=42, max_iter=300)
    model.fit(Xtr[keep], ytr[keep])

    return pd.Series(model.predict(X), index=net.index)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGES = ["Overview", "Store types", "Forecasting", "Anomalies", "SQL insights"]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Project:** Sales Demand Forecasting & Anomaly Detection\n\n"
    "**Data:** Corporación Favorita (Kaggle) — 1,115 stores x 942 days.\n\n"
    "**Model:** HistGradientBoosting, one-day-ahead with honest lags."
)

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("📊 Favorita Sales — Executive Overview")
    st.caption("Store-level daily sales for Corporación Favorita (2013-01-01 → 2015-07-31).")

    net = load_network()
    anom = load_anomalies()

    # --- KPI row -----------------------------------------------------------
    open_mask = net["sales"].notna()
    total_sales = net["sales"].sum()
    avg_daily = net["sales"].mean()
    with sqlite3.connect(DB_PATH) as conn:
        n_stores = conn.execute("SELECT COUNT(DISTINCT Store) FROM sales").fetchone()[0]
        promo_lift = conn.execute(
            "SELECT AVG(CASE WHEN Promo = 1 THEN Sales END) / "
            "AVG(CASE WHEN Promo = 0 THEN Sales END) * 100 - 100 "
            "FROM sales WHERE Open = 1"
        ).fetchone()[0]
    n_anom = int(anom["anomaly_z2"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total sales", f"${total_sales/1e6:,.0f}M")
    c2.metric("Avg daily sales", f"${avg_daily/1e6:,.2f}M")
    c3.metric("Stores", f"{n_stores:,}")
    c4.metric("Promo lift", f"+{promo_lift:.1f}%")
    c5.metric("Anomalous days (Z>2)", f"{n_anom}")

    st.markdown("---")

    # --- Network sales over time -------------------------------------------
    st.subheader("Network sales over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net["Date"], y=net["sales"],
                             name="Actual", line=dict(color=SALES_COLOR, width=2)))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Daily network sales ($)")
    st.plotly_chart(fig, width="stretch")

    # --- Weekly seasonality (average sales by day of week) ------------------
    st.subheader("Weekly seasonality — average sales by day of week")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    net["_dow"] = net["Date"].dt.dayofweek
    daily_avg = net.groupby("_dow")["sales"].mean().reindex(range(7))
    fig2 = go.Figure(go.Bar(x=dow_names, y=daily_avg.values,
                            marker_color=SALES_COLOR))
    fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                       yaxis_title="Average sales ($)")
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "**Why it matters:** Saturday is the busiest day and the weekend clearly "
        "dominates — that strong weekly rhythm is what the seasonal-naive baseline "
        "and the final model both exploit."
    )

# ============================================================================
# PAGE 2 — STORE TYPES
# ============================================================================
elif page == "Store types":
    st.title("🏪 Store Types & Promotions")
    st.caption("The network is 4 store types; type **a** is the revenue engine.")

    types = load_types()
    clean_meta = load_store_clean()

    # --- Revenue share by type ---------------------------------------------
    st.subheader("Sales breakdown by store type")
    share = types.groupby("store_type")["sales"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=share.index, y=share.values / 1e6, marker_color=SALES_COLOR,
                           text=[f"${v/1e9:.2f}B" for v in share.values]))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Store type", yaxis_title="Total sales ($M)")
    st.plotly_chart(fig, width="stretch")

    # --- Average sales & promo lift by type ---------------------------------
    st.subheader("Avg daily sales & promo lift by type")
    with sqlite3.connect(DB_PATH) as conn:
        agg = pd.read_sql_query(
            """SELECT StoreType,
                      ROUND(AVG(Sales), 0) AS AvgSales,
                      ROUND(AVG(Sales)/AVG(Customers), 2) AS SalesPerCustomer,
                      ROUND(AVG(CASE WHEN Promo=1 THEN Sales END) /
                            AVG(CASE WHEN Promo=0 THEN Sales END) * 100 - 100, 1)
                          AS PromoLiftPct,
                      COUNT(DISTINCT Store) AS Stores
               FROM sales_with_store
               WHERE Open = 1
               GROUP BY StoreType
               ORDER BY AvgSales DESC""", conn)
    agg["AvgSales"] = agg["AvgSales"] / 1e3
    agg.columns = ["Type", "Avg sales (k$)", "Sales per customer", "Promo lift %",
                   "Stores"]
    st.dataframe(agg, width="stretch")

    # --- Top 10 stores ------------------------------------------------------
    st.subheader("Top 10 stores by lifetime sales")
    cs = load_case_studies()
    top10 = (cs.groupby(["Store", "StoreType"])["Sales"].sum()
             .sort_values(ascending=False).head(10).reset_index())
    top10["Sales"] = top10["Sales"] / 1e6
    top10.columns = ["Store", "Type", "Lifetime sales ($M)"]
    fig2 = go.Figure(go.Bar(x=top10["Store"].astype(int), y=top10["Lifetime sales ($M)"],
                            marker_color=SALES_COLOR))
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="Store", yaxis_title="Lifetime sales ($M)")
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "**Why it matters:** store type drives the *level* of sales; the *weekly shape* "
        "is similar across types. That is why the project forecasts each type "
        "separately rather than one flat network curve."
    )

# ============================================================================
# PAGE 3 — FORECASTING
# ============================================================================
elif page == "Forecasting":
    st.title("🔮 Forecasting — HistGradientBoosting")
    st.caption(
        "Final model (Phase 5): one-day-ahead forecast using **only information "
        "known the day before** (lags, rolling stats, promo, calendar). "
        "Out-of-sample numbers: **validation MAE $0.335M · test MAE $0.283M** — an "
        "~84–86% reduction vs the seasonal-naive baseline ($2.01M)."
    )

    net = load_network()

    saved = load_model()
    model, features = saved["model"], saved["features"]
    X = build_feature_frame(net, features)
    mask = ~X.isna().any(axis=1)                 # warm-up rows have no lag history

    # Two honest forecast sources:
    #  - the SAVED model (trained on train + validation) for the TEST window
    #  - a re-fit TRAIN-ONLY model for the VALIDATION window (the saved model
    #    is in-sample there, so it cannot honestly score it)
    saved_pred = pd.Series(model.predict(X).astype(float), index=net.index)
    to_pred = train_only_forecast()

    # --- Full-series chart: saved model's view over the whole history ---------
    st.subheader("Actual vs saved model — full series")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net["Date"], y=net["sales"],
                             name="Actual", line=dict(color=SALES_COLOR, width=1.5)))
    fig.add_trace(go.Scatter(x=net["Date"], y=saved_pred,
                             name="Saved model", line=dict(color=PRED_COLOR, width=1.5)))
    fig.add_vline(x=pd.Timestamp("2015-07-01"), line_dash="dot", line_color="gray",
                  annotation_text="out-of-sample →",
                  annotation_position="top right")
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Daily sales ($)")
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "**Honesty note — do not over-read the chart.** The saved model was trained "
        "on train + validation, so the orange line is *in-sample* for everything "
        "before 2015-07-16. Only the **Test** block is truly out-of-sample for it. "
        "The honest validation numbers come from the train-only model shown next."
    )

    # --- Honest out-of-sample comparison --------------------------------------
    st.subheader("Honest out-of-sample: validation & test (Jul 2015)")

    # Build one chart where each window is scored by a model that never saw it.
    holdout_rows = []
    for label, lo, hi, pred_src in [
        ("Validation\n(train-only model)", "2015-07-01", "2015-07-15", to_pred),
        ("Test\n(saved model)", "2015-07-16", "2015-07-31", saved_pred),
    ]:
        sub_idx = (net["Date"] >= lo) & (net["Date"] <= hi)
        holdout_rows.append(pd.DataFrame({
            "Date": net.loc[sub_idx, "Date"],
            "actual": net.loc[sub_idx, "sales"],
            "pred": pred_src[sub_idx],
            "window": label,
        }))
    holdout = pd.concat(holdout_rows)

    fig2 = go.Figure()
    for label in ["Validation\n(train-only model)", "Test\n(saved model)"]:
        part = holdout[holdout["window"] == label]
        fig2.add_trace(go.Scatter(x=part["Date"], y=part["actual"], name=f"{label} · actual",
                                  line=dict(color=SALES_COLOR, width=2)))
        fig2.add_trace(go.Scatter(x=part["Date"], y=part["pred"],
                                  name=f"{label} · predicted",
                                  line=dict(color=PRED_COLOR, width=2, dash="dash")))
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                       yaxis_title="Daily sales ($)")
    st.plotly_chart(fig2, width="stretch")

    # --- Error metrics per window (model that never saw that window) ----------
    for label in ["Validation\n(train-only model)", "Test\n(saved model)"]:
        part = holdout[holdout["window"] == label]
        mae = (part["actual"] - part["pred"]).abs().mean() / 1e6
        rmse = np.sqrt(((part["actual"] - part["pred"]) ** 2).mean()) / 1e6
        c = st.columns(2)
        c[0].metric(f"{label} MAE", f"${mae:.3f}M")
        c[1].metric(f"{label} RMSE", f"${rmse:.3f}M")

    st.info(
        "**Why two models on this page?** The project's *final* model (saved to "
        "`final_model.pkl`) was retrained on train + validation — standard practice "
        "so the test results are the real headline. That makes it in-sample on "
        "validation, so to honestly score the validation window we re-fit an "
        "identical model on train-only data (the exact protocol in notebook 05). "
        "This page shows each window scored by a model that never saw it."
    )

    # --- Feature importance -------------------------------------------------
    st.subheader("What drives the forecast? Permutation importance")
    with st.spinner("Computing permutation importance (a few seconds)…"):
        @st.cache_data(show_spinner=False)
        def compute_importances():
            tr_feats = build_feature_frame(net, features)
            tr = net.copy()
            tr["_y"] = net["sales"]
            mask = ~tr_feats.isna().any(axis=1)
            Xt = tr_feats[mask]
            yt = tr.loc[mask.values, "sales"]
            perm = permutation_importance(model, Xt, yt, n_repeats=3,
                                          random_state=42, scoring="neg_mean_absolute_error")
            idx = np.argsort(perm.importances_mean)
            return pd.DataFrame({
                "Feature": np.array(features)[idx],
                "Importance": perm.importances_mean[idx],
            })

        imp = compute_importances()
    imp = imp.sort_values("Importance", ascending=True)
    fig3 = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"],
                            orientation="h", marker_color=PRED_COLOR))
    fig3.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="Importance (Δ MAE $)")
    st.plotly_chart(fig3, width="stretch")

    st.info(
        "**Why it matters:** `sales_lag_7` and `rolling_mean_7` dominate — the model "
        "learned the weekly rhythm, exactly what EDA predicted. Promo and weekend "
        "features add the second wave of signal that beats every naive baseline."
    )

# ============================================================================
# PAGE 4 — ANOMALIES
# ============================================================================
elif page == "Anomalies":
    st.title("🚨 Anomaly Detection")
    st.caption(
        "Statistical baseline (Phase 6): rolling 7-day mean/std + Z-score on the "
        "model's residual. An anomaly = day whose sales **differ from the expected "
        "pattern**, not an error. 121 of 942 days flagged at Z > 2."
    )

    anom = load_anomalies()
    anom["Date"] = pd.to_datetime(anom["Date"])
    anom["anomaly_z2"] = anom["anomaly_z2"].astype(bool)

    # --- Time series with anomaly markers -----------------------------------
    st.subheader("Daily sales with anomalous days highlighted")
    anom_pts = anom[anom["anomaly_z2"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anom["Date"], y=anom["actual"],
                             name="Sales", line=dict(color=SALES_COLOR, width=1.5)))
    fig.add_trace(go.Scatter(x=anom_pts["Date"], y=anom_pts["actual"],
                             name="Anomaly (Z>2)", mode="markers",
                             marker=dict(color=ANOM_COLOR, size=6)))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Daily network sales ($)")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "The Jul-2015 tail is the test window: those first two test days (Jul 16–17) "
        "were flagged anomalous — sales dropped well below the model's expectation."
    )

    # --- Duration sliders to explore anomaly flags ---------------------------
    st.subheader("Explore a date range")
    c1, c2 = st.columns(2)
    lo = pd.to_datetime(c1.date_input("From", anom["Date"].min()))
    hi = pd.to_datetime(c2.date_input("To", anom["Date"].max()))
    sub = anom[(anom["Date"] >= lo) & (anom["Date"] <= hi)]

    tab_pts, tab_res, tab_z = st.tabs(["Anomalies in range", "Residuals", "Z-scores"])
    with tab_pts:
        flagged = sub[sub["anomaly_z2"]]
        if len(flagged) == 0:
            st.write("No anomalous days in this range.")
        else:
            disp = flagged[["Date", "actual", "pred", "residual", "z_score"]].copy()
            disp["Date"] = disp["Date"].dt.date
            disp.columns = ["Date", "Actual ($)", "Predicted ($)", "Residual ($)", "Z"]
            st.dataframe(disp, width="stretch")

    with tab_res:
        fig2 = go.Figure(go.Scatter(x=sub["Date"], y=sub["residual"],
                                    mode="markers", marker=dict(color=SALES_COLOR, size=4)))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Residual (actual − predicted, $)")
        st.plotly_chart(fig2, width="stretch")

    with tab_z:
        fig3 = go.Figure(go.Scatter(x=sub["Date"], y=sub["z_score"],
                                    mode="markers", marker=dict(color=PRED_COLOR, size=4)))
        fig3.add_hline(y=2, line_dash="dash", line_color=ANOM_COLOR,
                       annotation_text="Z = +2", annotation_position="top right")
        fig3.add_hline(y=-2, line_dash="dash", line_color=ANOM_COLOR,
                       annotation_text="Z = −2", annotation_position="bottom right")
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Z-score")
        st.plotly_chart(fig3, width="stretch")

    anom_summary = pd.DataFrame({
        "Metric": ["Days monitored", "Days flagged (Z>2)", "Flag rate (%)",
                   "Avg absolute residual ($)"],
        "Value": [len(anom), int(anom["anomaly_z2"].sum()),
                  round(anom["anomaly_z2"].mean() * 100, 1),
                  round(anom["residual"].abs().mean())],
    })
    st.dataframe(anom_summary, width="stretch")

    st.info(
        "**Why it matters:** a few explainable spikes (payday, Christmas) are expected. "
        "This detection surfaces the *unexplained* deviations — the ones worth a "
        "human look — which is exactly what a business analyst wants from a "
        "monitoring dashboard."
    )

# ============================================================================
# PAGE 5 — SQL INSIGHTS
# ============================================================================
else:  # "SQL insights"
    st.title("🗄️ SQL Business Insights")
    st.caption(
        "Every number on this page is a live SQL query against "
        "`data/processed/sales.db` (Phase 7). The queries live in "
        "`sql/analysis.sql` so the whole analysis is reproducible and auditable."
    )

    def run_sql(query: str):
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn)

    # --- Executive summary --------------------------------------------------
    st.subheader("Executive summary")
    exec_query = """
        SELECT 'Total sales' AS Metric, printf('$%,.0f', SUM(Sales)) AS Value
        FROM sales_with_store WHERE Open = 1
        UNION ALL
        SELECT 'Avg daily sales', printf('$%,.0f', AVG(Sales))
        FROM sales_with_store WHERE Open = 1
        UNION ALL
        SELECT 'Total store-days open', printf('%,d', COUNT(*))
        FROM sales_with_store WHERE Open = 1
        UNION ALL
        SELECT 'Number of stores', printf('%d', COUNT(DISTINCT Store))
        FROM sales_with_store
        UNION ALL
        SELECT 'Promo lift', printf('+%.1f%%', AVG(CASE WHEN Promo=1 THEN Sales END)/
               AVG(CASE WHEN Promo=0 THEN Sales END)*100-100)
        FROM sales_with_store WHERE Open = 1
        UNION ALL
        SELECT 'Anomalous days (Z>2)', printf('%d', COUNT(*))
        FROM anomalies WHERE anomaly_z2 = 1
    """
    st.dataframe(run_sql(exec_query), width="stretch", hide_index=True)

    # --- Store type breakdown -----------------------------------------------
    st.subheader("Sales by store type")
    st.dataframe(run_sql(
        """SELECT StoreType,
                  COUNT(DISTINCT Store) AS Stores,
                  printf('$%,.0f', SUM(Sales)) AS TotalSales,
                  ROUND(SUM(Sales)*100.0/(SELECT SUM(Sales) FROM sales_with_store
                                          WHERE Open=1), 2) AS Pct_network,
                  ROUND(AVG(Sales),0) AS AvgDailySales,
                  ROUND(AVG(Customers),0) AS AvgDailyCustomers
           FROM sales_with_store WHERE Open=1 GROUP BY StoreType
           ORDER BY TotalSales DESC"""),
        width="stretch", hide_index=True)

    # --- Top / bottom stores ------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 stores")
        st.dataframe(run_sql(
            """SELECT s.Store, st.StoreType, printf('$%,.0f', SUM(s.Sales)) AS TotalSales
               FROM sales s JOIN stores st ON s.Store=st.Store
               WHERE s.Open=1 GROUP BY s.Store ORDER BY TotalSales DESC LIMIT 5"""),
            width="stretch", hide_index=True)
    with col2:
        st.subheader("Bottom 5 stores")
        st.dataframe(run_sql(
            """SELECT s.Store, st.StoreType, printf('$%,.0f', SUM(s.Sales)) AS TotalSales
               FROM sales s JOIN stores st ON s.Store=st.Store
               WHERE s.Open=1 GROUP BY s.Store ORDER BY TotalSales ASC LIMIT 5"""),
            width="stretch", hide_index=True)

    # --- Monthly trend ------------------------------------------------------
    st.subheader("Monthly sales trend")
    monthly = run_sql(
        """SELECT substr(Date,1,7) AS Month,
                  ROUND(SUM(Sales)/1e6,1) AS Sales_M
           FROM sales_with_store WHERE Open=1
           GROUP BY Month ORDER BY Month""")
    fig = go.Figure(go.Bar(x=monthly["Month"], y=monthly["Sales_M"],
                           marker_color=SALES_COLOR))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Month", yaxis_title="Sales ($M)")
    st.plotly_chart(fig, width="stretch")

    # --- Day of week --------------------------------------------------------
    st.subheader("Average sales by day of week")
    dow = run_sql(
        """SELECT DayOfWeek,
                  ROUND(AVG(Sales),0) AS AvgSales
           FROM sales_with_store WHERE Open=1
           GROUP BY DayOfWeek ORDER BY DayOfWeek""")
    names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    dow["Day"] = dow["DayOfWeek"].map(names)
    fig2 = go.Figure(go.Bar(x=dow["Day"], y=dow["AvgSales"] / 1e6,
                            marker_color=SALES_COLOR))
    fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                       yaxis_title="Avg daily sales ($M)")
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "**Why SQL is in the project:** data is rarely delivered only as CSVs. This "
        "page shows the same facts can be produced with a real database and plain "
        "SQL — a skill recruiters look for — with the full script in `sql/analysis.sql`."
    )