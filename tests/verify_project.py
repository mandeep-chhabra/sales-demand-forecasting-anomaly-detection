"""
verify_project.py — end-to-end verification of the Sales Forecasting project
=============================================================================

Run from anywhere with:

    python tests/verify_project.py

It checks, in order:
  1. The project tree is intact (required folders and files exist).
  2. Cleaned data passes integrity checks (shapes, dtypes, no nulls in
     critical columns, split boundaries).
  3. The Phase-3 features were built without leakage (lag columns reference
     only past actuals).
  4. The final model loads and scores sensibly on the holdout.
  5. The anomaly table is consistent with the network series.
  6. The SQLite database is queryable and answers match Phase-7 numbers.
  7. The Streamlit dashboard runs through Streamlit's AppTest harness.

If every check reaches "PASS", the whole pipeline is reproducible and the
dashboard is launchable from the artifacts alone.
"""

import os
import sys
import pickle
import sqlite3

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(PROJECT_ROOT, "data", "processed")
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
NB = os.path.join(PROJECT_ROOT, "notebooks")
DB_PATH = os.path.join(PROC, "sales.db")
MODEL_PATH = os.path.join(PROC, "final_model.pkl")

# ---------------------------------------------------------------------------
# Tiny test harness: a list of (name, callable) checks
# ---------------------------------------------------------------------------
_CHECKS = []


def check(name):
    def deco(fn):
        _CHECKS.append((name, fn))
        return fn
    return deco


def _run_all():
    passed = 0
    failed = []
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 — we want to catch everything
            print(f"[FAIL] {name}\n       -> {exc}")
            failed.append(name)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(_CHECKS)} checks passed")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
    print("ALL CHECKS PASSED — project is healthy.")


# ===========================================================================
# 1. PROJECT TREE
# ===========================================================================
@check("project tree — required folders/files exist")
def tree():
    required_files = [
        os.path.join(NB, "01_data_understanding.ipynb"),
        os.path.join(NB, "02_eda.ipynb"),
        os.path.join(NB, "03_feature_engineering.ipynb"),
        os.path.join(NB, "04_baseline_forecasting.ipynb"),
        os.path.join(NB, "05_final_forecast_model.ipynb"),
        os.path.join(NB, "06_anomaly_detection.ipynb"),
        os.path.join(NB, "07_sql_analysis.ipynb"),
        os.path.join(PROJECT_ROOT, "app", "dashboard.py"),
        os.path.join(PROJECT_ROOT, "sql", "analysis.sql"),
        os.path.join(PROC, "final_model.pkl"),
        DB_PATH,
    ]
    missing = [p for p in required_files if not os.path.exists(p)]
    assert not missing, f"Missing: {missing}"


@check("notebooks were executed (have outputs)")
def notebooks_executed():
    import json
    for name in ["01_data_understanding", "02_eda", "03_feature_engineering",
                 "04_baseline_forecasting", "05_final_forecast_model",
                 "06_anomaly_detection", "07_sql_analysis"]:
        with open(os.path.join(NB, name + ".ipynb"), encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code_cells, f"{name}: no code cells"
        for c in code_cells:
            assert c.get("execution_count") is not None and c.get("outputs"), \
                f"{name}: cell not executed"


# ===========================================================================
# 2. CLEANED DATA INTEGRITY
# ===========================================================================
@check("train_clean: shape and no nulls")
def train_clean():
    df = pd.read_csv(os.path.join(PROC, "train_clean.csv"))
    assert df.shape == (1_017_209, 9), f"shape {df.shape}"
    assert df.isna().sum().sum() == 0, "train_clean has nulls"
    assert set(df["StateHoliday"].unique()) <= {"none", "a", "b", "c"}, \
        f"StateHoliday values {sorted(df['StateHoliday'].unique())}"
    # closed days are zero-sales days, never dropped
    closed = (df["Open"] == 0)
    assert (df.loc[closed, "Sales"] == 0).all(), "closed days must have Sales=0"


@check("store_clean: shape and no nulls")
def store_clean():
    df = pd.read_csv(os.path.join(PROC, "store_clean.csv"))
    assert df.shape == (1115, 11), f"shape {df.shape}"
    assert df.isna().sum().sum() == 0, "store_clean has nulls"
    assert df["Store"].is_unique


@check("temporal split is non-overlapping and covers full series")
def splits():
    net = pd.read_csv(os.path.join(PROC, "network_features.csv"),
                      parse_dates=["Date"])
    splits = [("train", "2013-01-01", "2015-06-30"),
              ("validation", "2015-07-01", "2015-07-15"),
              ("test", "2015-07-16", "2015-07-31")]
    for label, lo, hi in splits:
        sub = net[net["split"] == label]
        assert sub["Date"].min() == pd.Timestamp(lo), f"{label} min {sub['Date'].min()}"
        assert sub["Date"].max() == pd.Timestamp(hi), f"{label} max {sub['Date'].max()}"
        assert sub["Date"].is_monotonic_increasing
    # no overlap between consecutive splits
    assert net.loc[net["split"] == "train", "Date"].max() < \
        net.loc[net["split"] == "validation", "Date"].min()
    assert net.loc[net["split"] == "validation", "Date"].max() < \
        net.loc[net["split"] == "test", "Date"].min()


# ===========================================================================
# 3. NO-LEAKAGE CHECK ON LAG/BOLTING FEATURES
# ===========================================================================
@check("lag features reference only past actuals (no leakage)")
def no_leakage():
    net = pd.read_csv(os.path.join(PROC, "network_features.csv"),
                      parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sales = net["sales"].values
    for lag in [1, 2, 7, 14, 28]:
        # first valid lag value must equal the actual `lag` days earlier
        i = lag
        np.testing.assert_allclose(net.loc[i, f"sales_lag_{lag}"], sales[i - lag])
    # rolling_mean_7 at index i uses sales[i-7:i]
    np.testing.assert_allclose(
        net.loc[7, "rolling_mean_7"], np.mean(sales[0:7]))
    assert net["sales_lag_1"].iloc[:1].isna().all(), "first row lag should be NaN"


# ===========================================================================
# 4. MODEL
# ===========================================================================
@check("final model loads and predicts out-of-sample")
def model():
    saved = joblib.load(MODEL_PATH)
    assert saved["kind"] == "ml"
    model, features = saved["model"], saved["features"]
    assert len(features) == 32, f"features {len(features)}"

    net = pd.read_csv(os.path.join(PROC, "network_features.csv"),
                      parse_dates=["Date"])

    # rebuild features exactly as notebook 05 does
    drop_cols = ["Date", "sales", "split", "n_open", "year", "state_holiday"]
    base = [c for c in net.columns if c not in drop_cols]
    dummies = pd.get_dummies(net["state_holiday"], prefix="hol", dtype=int)
    full = pd.concat([net[base].reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)
    X = full[[c for c in features if c in full.columns]]

    test_mask = net["split"] == "test"
    X_test, y_test = X[test_mask], net.loc[test_mask, "sales"]
    assert not X_test.isna().any().any(), "test features have NaN"
    pred = model.predict(X_test)
    mae = float(np.abs(pred - y_test.values).mean()) / 1e6
    # Phase 5 reported test MAE ≈ $0.283M; allow a small tolerance band so the
    # check is robust, but a big regression would fail loudly.
    assert mae < 0.45, f"test MAE ${mae:.3f}M is far from the reported $0.283M"


# ===========================================================================
# 5. ANOMALIES
# ===========================================================================
@check("anomaly table is consistent with the network series")
def anomalies():
    net = pd.read_csv(os.path.join(PROC, "network_features.csv"),
                      parse_dates=["Date"])
    anom = pd.read_csv(os.path.join(PROC, "anomaly_scores.csv"),
                       parse_dates=["Date"])
    assert len(anom) == len(net) == 942, f"lengths {len(anom)} vs {len(net)}"
    assert (anom["Date"].values == net["Date"].values).all(), "dates out of sync"
    n_flagged = int(anom["anomaly_z2"].sum())
    assert 100 <= n_flagged <= 140, f"unexpected flag count {n_flagged}"
    # test window matches anomaly_test.csv
    atest = pd.read_csv(os.path.join(PROC, "anomaly_test.csv"), parse_dates=["Date"])
    tail = anom[anom["Date"] >= "2015-07-16"]
    assert len(tail) == len(atest), "anomaly_test mismatch"


# ===========================================================================
# 6. SQLITE DATABASE
# ===========================================================================
@check("SQLite database has expected tables/rows and answers match")
def sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"sales", "stores", "anomalies"} <= tables, f"tables {tables}"
        n_sales = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        assert n_sales == 1_017_209, f"sales rows {n_sales}"
        n_stores = conn.execute("SELECT COUNT(*) FROM stores").fetchone()[0]
        assert n_stores == 1115, f"stores rows {n_stores}"
        n_anom = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE anomaly_z2 = 1").fetchone()[0]
        assert n_anom == int(pd.read_csv(os.path.join(PROC, "anomaly_scores.csv"),
                                         usecols=["anomaly_z2"])["anomaly_z2"].sum())
        # promo lift ≈ +38.8% (documented in EDA)
        lift = conn.execute(
            "SELECT AVG(CASE WHEN Promo=1 THEN Sales END) / "
            "AVG(CASE WHEN Promo=0 THEN Sales END) * 100 - 100 "
            "FROM sales WHERE Open=1").fetchone()[0]
        assert 35 <= lift <= 45, f"promo lift {lift}"
    finally:
        conn.close()


# ===========================================================================
# 7. DASHBOARD SMOKE TEST
# ===========================================================================
@check("dashboard launches and all pages render")
def dashboard():
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    from streamlit.testing.v1 import AppTest
    app = os.path.join(PROJECT_ROOT, "app", "dashboard.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.run()
    assert len(at.exception) == 0, [e.value for e in at.exception]
    for page in ["Store types", "Forecasting", "Anomalies", "SQL insights",
                 "Overview"]:
        at.sidebar.radio[0].set_value(page).run()
        assert len(at.exception) == 0, \
            f"page '{page}': {[e.value for e in at.exception]}"


if __name__ == "__main__":
    _run_all()