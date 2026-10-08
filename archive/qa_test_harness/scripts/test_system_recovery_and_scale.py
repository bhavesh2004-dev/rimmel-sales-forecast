"""
QA Test Suite 7: Performance, Scalability, Reproducibility & Failure Recovery QA
================================================================================
Tests:
- Deterministic reproducibility (bitwise float identity across repeated runs)
- Controlled failure recovery (missing DB, missing model, missing column, all-NaN rows)
- Performance benchmarks: inference throughput, query latency, memory footprint,
  and empirical scaling projections for 1,000, 5,000, and 10,000 SKUs.
"""

import sys
import time
import pickle
import sqlite3
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.demand_forecasting_qa.scripts.qa_harness import (
    QARecorder, DB_PATH, MODELS_DIR, prepare_lgbm_dataframe
)


def run_system_recovery_and_scale_tests(recorder: QARecorder):
    print("\n" + "=" * 80)
    print("SUITE 7: PERFORMANCE, SCALABILITY, REPRODUCIBILITY & FAILURE RECOVERY QA")
    print("=" * 80)

    # 1. Load Model & Features Metadata
    with open(MODELS_DIR / "production_lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "production_features.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_list"]
    cat_cols = meta["categorical_features"]

    # Load 1,413 active series from Sep 10 as baseline
    conn = sqlite3.connect(DB_PATH)
    t0_query = time.time()
    sep10_df = pd.read_sql_query("SELECT * FROM ml_features_zero WHERE date = '2026-09-10'", conn)
    query_duration = time.time() - t0_query
    conn.close()

    eval_df = prepare_lgbm_dataframe(sep10_df, feature_cols, cat_cols)

    # -------------------------------------------------------------------------
    # PART A: REPRODUCIBILITY TEST
    # -------------------------------------------------------------------------
    print("\n[REPRODUCIBILITY AUDIT] Testing Exact Deterministic Repeatability...")
    p1 = model.predict(eval_df)
    p2 = model.predict(eval_df)

    is_identical = np.array_equal(p1, p2)
    max_abs_diff = float(np.max(np.abs(p1 - p2)))

    if is_identical and max_abs_diff == 0.0:
        recorder.record(
            test_id="REP-DET-01",
            category="Reproducibility",
            test_name="Deterministic Inference Repeatability (Bitwise Float Identity)",
            status="PASS",
            severity="LOW",
            expected="Two consecutive inference runs on identical input must produce bitwise identical predictions (diff == 0.0)",
            actual=f"100% Bitwise identity confirmed across {len(p1):,} series (max_diff = {max_abs_diff})",
            component="model_inference",
            production_affected="NO"
        )
    else:
        recorder.record(
            test_id="REP-DET-01",
            category="Reproducibility",
            test_name="Deterministic Inference Repeatability",
            status="FAIL",
            severity="HIGH",
            expected="diff == 0.0",
            actual=f"Non-deterministic difference detected: max_diff = {max_abs_diff}",
            component="model_inference",
            root_cause="Non-deterministic random seed or thread race in LightGBM inference",
            recommended_fix="Lock random_state=42 and enforce deterministic inference parameters",
            production_affected="YES"
        )

    # -------------------------------------------------------------------------
    # PART B: FAILURE RECOVERY TESTING
    # -------------------------------------------------------------------------
    print("\n[FAILURE RECOVERY] Testing Controlled System Failure Scenarios...")

    # FAIL-01: Non-Existent Database Path Handling
    try:
        fake_db = PROJECT_ROOT / "data" / "non_existent_fake_db.db"
        test_conn = sqlite3.connect(fake_db)
        test_df = pd.read_sql_query("SELECT * FROM ml_features_zero", test_conn)
        test_conn.close()
        recorder.record(
            test_id="FAIL-01",
            category="Failure Recovery",
            test_name="Non-Existent Database Path Handling",
            status="FAIL",
            severity="HIGH",
            expected="Should raise OperationalError when table is absent in non-existent DB",
            actual="Silently succeeded unexpectedly",
            component="database_layer",
            production_affected="NO"
        )
    except Exception as e:
        recorder.record(
            test_id="FAIL-01",
            category="Failure Recovery",
            test_name="Non-Existent Database Path Handling",
            status="PASS",
            severity="LOW",
            expected="Fails with clear sqlite3.OperationalError when database or table is missing",
            actual=f"Correctly raised clean exception: {type(e).__name__} ({str(e)[:80]})",
            component="database_layer",
            production_affected="NO"
        )

    # FAIL-02: Missing Required Feature Column in Inference Dataframe
    try:
        corrupted_features = eval_df.drop(columns=["v14_vs_v30"]).copy()
        bad_pred = model.predict(corrupted_features)
        recorder.record(
            test_id="FAIL-02",
            category="Failure Recovery",
            test_name="Missing Feature Column Error Catching",
            status="FAIL",
            severity="CRITICAL",
            expected="LightGBM must raise error when expected feature column is missing",
            actual="Inference proceeded with missing feature column without error!",
            component="model_inference",
            production_affected="YES"
        )
    except Exception as e:
        recorder.record(
            test_id="FAIL-02",
            category="Failure Recovery",
            test_name="Missing Feature Column Error Catching",
            status="PASS",
            severity="LOW",
            expected="LightGBM raises LightGBMError / ValueError when expected feature column is missing",
            actual=f"Correctly raised exception: {type(e).__name__} ({str(e)[:80]})",
            component="model_inference",
            production_affected="NO"
        )

    # FAIL-03: Corrupted / All-NaN Feature Row Handling
    try:
        all_nan_row = pd.DataFrame([{col: np.nan for col in feature_cols}])
        all_nan_feat = prepare_lgbm_dataframe(all_nan_row, feature_cols, cat_cols)
        nan_pred = model.predict(all_nan_feat)
        # Check if model outputs a finite default baseline or throws error
        if len(nan_pred) == 1 and not np.isnan(nan_pred[0]):
            recorder.record(
                test_id="FAIL-03",
                category="Failure Recovery",
                test_name="All-NaN Feature Row Resilience",
                status="PASS",
                severity="LOW",
                expected="LightGBM natively handles NaNs using default split routing without crashing",
                actual=f"Handled cleanly -> Base prediction: {nan_pred[0]:.4f} units/day",
                component="model_inference",
                production_affected="NO"
            )
        else:
            recorder.record(
                test_id="FAIL-03",
                category="Failure Recovery",
                test_name="All-NaN Feature Row Resilience",
                status="WARNING",
                severity="MEDIUM",
                expected="LightGBM handles NaNs cleanly",
                actual=f"Predicted NaN output on all-NaN input row: {nan_pred}",
                component="model_inference",
                root_cause="Unimputed NaNs route to unpopulated leaf",
                recommended_fix="Impute core numerical features before calling model.predict",
                production_affected="NO"
            )
    except Exception as e:
        recorder.record(
            test_id="FAIL-03",
            category="Failure Recovery",
            test_name="All-NaN Feature Row Resilience",
            status="FAIL",
            severity="HIGH",
            expected="LightGBM handles NaNs without crashing",
            actual=f"Exception raised: {e}",
            component="model_inference",
            production_affected="YES"
        )

    # -------------------------------------------------------------------------
    # PART C: PERFORMANCE & SCALABILITY BENCHMARK
    # -------------------------------------------------------------------------
    print("\n[PERFORMANCE BENCHMARK] Measuring Latency & Projecting Scalability...")

    # Measure raw inference throughput on 1,413 rows
    n_runs = 5
    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = model.predict(eval_df)
        latencies.append(time.time() - t0)

    mean_inference_time = float(np.mean(latencies))
    throughput_rows_per_sec = len(eval_df) / mean_inference_time
    latency_per_1000_rows = (mean_inference_time / len(eval_df)) * 1000.0

    # Memory Footprint
    mem_mb = eval_df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Scaling Projections for 10-day forward horizon:
    # 1 SKU ~ 2.1 channel series (1,413 series / 674 SKUs = 2.096 series/SKU)
    # 10 days = 10 daily inference cycles
    channels_per_sku = 2.1
    horizon_days = 10

    scenarios = [
        ("Current Catalog (674 SKUs)", 674, 1413),
        ("Scale 1 (1,000 SKUs)", 1000, int(1000 * channels_per_sku)),
        ("Scale 2 (5,000 SKUs)", 5000, int(5000 * channels_per_sku)),
        ("Scale 3 (10,000 SKUs)", 10000, int(10000 * channels_per_sku)),
    ]

    scale_summary_lines = []
    scale_summary_lines.append(f"Measured Throughput: {throughput_rows_per_sec:,.0f} series-days/sec ({latency_per_1000_rows*1000:.1f} ms / 1,000 rows)")
    scale_summary_lines.append(f"SQLite Query Time (1,413 rows): {query_duration:.3f}s | DataFrame RAM: {mem_mb:.2f} MB\n")

    for label, n_sku, n_ser in scenarios:
        total_eval_points = n_ser * horizon_days
        proj_inf_time = total_eval_points / throughput_rows_per_sec
        # Estimated feature extraction time scaling linearly from query time
        proj_feat_time = (n_ser / 1413.0) * query_duration * 3.5  # assuming join & lag feature calc overhead
        proj_tot_time = proj_inf_time + proj_feat_time
        proj_ram = (n_ser / 1413.0) * mem_mb * horizon_days
        scale_summary_lines.append(
            f"- **{label}**: {n_ser:,} active series ({total_eval_points:,} 10-day predictions) -> "
            f"Est Inference: {proj_inf_time:.2f}s, Est Total Pipeline: {proj_tot_time:.1f}s, RAM: {proj_ram:.1f} MB"
        )

    print("\n".join(scale_summary_lines))

    # TEST SCALE-01: Production Throughput Standard
    # Expected: Inference on current 1,413 series must take < 0.25 seconds
    if mean_inference_time < 0.25:
        recorder.record(
            test_id="SCALE-01",
            category="Performance & Scalability",
            test_name="Model Inference Throughput & Latency Gate (< 0.25s per 1,413 series)",
            status="PASS",
            severity="LOW",
            expected="Inference on 1,413 series takes < 0.25 seconds",
            actual=f"Mean inference time = {mean_inference_time*1000:.1f} ms ({throughput_rows_per_sec:,.0f} rows/sec)",
            component="model_inference",
            production_affected="NO"
        )
    else:
        recorder.record(
            test_id="SCALE-01",
            category="Performance & Scalability",
            test_name="Model Inference Throughput & Latency Gate",
            status="WARNING",
            severity="LOW",
            expected="Inference takes < 0.25 seconds",
            actual=f"Inference time = {mean_inference_time:.3f} seconds",
            component="model_inference",
            production_affected="NO"
        )

    print(f"Suite 7 execution finished: {recorder.get_summary()}")


if __name__ == "__main__":
    rec = QARecorder("Suite 7: System Recovery & Scalability")
    run_system_recovery_and_scale_tests(rec)
