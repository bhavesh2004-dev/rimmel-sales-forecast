"""
QA Test Suite 2: Demand Dynamics & Prediction Sanity Testing
============================================================
Evaluates model behavior across 18 distinct demand archetypes:
1. Completely zero-demand SKU
2. Long zero-demand period
3. One isolated sale after long zero period
4. Sudden spike (10x normal)
5. Multiple consecutive spikes
6. Sudden demand collapse
7. Gradual demand increase
8. Gradual demand decrease
9. Highly volatile SKU (CV > 2.5)
10. Stable high-volume SKU (CV < 0.2)
11. Stable low-volume SKU
12. Intermittent SKU
13. Demand only on weekends
14. Demand only on weekdays
15. Extremely high-volume SKU
16. Extremely low-volume SKU
17. SKU with one giant transaction
18. SKU with repeated small transactions

Saves outputs to: test/demand_forecasting_qa/outputs/edge_case_predictions.csv
"""

import sys
import pickle
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.demand_forecasting_qa.scripts.qa_harness import (
    QARecorder, DB_PATH, MODELS_DIR, OUTPUTS_DIR, GENERATED_DATA_DIR, prepare_lgbm_dataframe
)


def run_demand_dynamics_tests(recorder: QARecorder):
    print("\n" + "=" * 80)
    print("SUITE 2: DEMAND DYNAMICS & PREDICTION SANITY TESTING (18 ARCHETYPES)")
    print("=" * 80)

    import json

    # 1. Load Model & Features
    with open(MODELS_DIR / "production_lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "production_features.json", "r", encoding="utf-8") as f:
        features_meta = json.load(f)
    feature_cols = features_meta["feature_list"]
    cat_cols = features_meta["categorical_features"]

    # Query template row from ml_features_zero as baseline template
    conn = sqlite3.connect(DB_PATH)
    template_df = pd.read_sql_query("SELECT * FROM ml_features_zero WHERE date = '2026-09-10' LIMIT 1", conn)
    conn.close()

    base_row = template_df[feature_cols].iloc[0].to_dict()

    edge_case_records = []

    # Helper function to construct synthetic feature row and infer
    def evaluate_archetype(
        archetype_id: str,
        archetype_name: str,
        modifications: dict,
        expected_bounds: tuple,  # (min_plausible, max_plausible)
        test_code: str
    ):
        row_dict = base_row.copy()
        row_dict.update(modifications)
        test_df = pd.DataFrame([row_dict])
        test_feat = prepare_lgbm_dataframe(test_df, feature_cols, cat_cols)

        # 1. Raw Inference
        raw_pred = float(np.clip(model.predict(test_feat)[0], 0, None))

        # 2. Calibration Logic (Exp6)
        v7 = row_dict.get("v7", 0.0)
        v14 = row_dict.get("v14", 0.0)
        v30 = row_dict.get("v30", 0.0)
        promo = row_dict.get("promo_days_30", 0.0)
        traffic_mom = row_dict.get("amazon_sessions_momentum", 1.0)
        in_stock = row_dict.get("in_stock_flag", 1)
        has_inv = row_dict.get("has_inventory_signal", 1)

        z_calib = (v7 == 0 and v14 == 0 and v30 == 0 and promo == 0 and traffic_mom <= 1.25)
        stk_calib = (in_stock == 0 and has_inv == 1)

        calib_pred = raw_pred
        if z_calib:
            calib_pred *= 0.10
        if stk_calib:
            calib_pred *= 0.10

        # Confidence & Risk Classification
        stock_val = row_dict.get("current_stock", 999.0)
        cv = row_dict.get("cv_30", 0.5)
        v90 = row_dict.get("v90", 0.0)

        if in_stock == 0 or stock_val == 0:
            conf = "Low (Stockout Suppressed)"
            risk = "STOCKOUT RISK"
        elif cv > 1.2:
            conf = "Low (Volatile Demand)"
            risk = "VOLATILITY MONITORING"
        elif calib_pred >= 0.5 and cv <= 0.6:
            conf = "High (Stable Continuous)"
            risk = "NORMAL HEALTHY"
        elif calib_pred < 0.1 and v90 < 0.1:
            conf = "High (Confirmed Sparse/Zero)"
            risk = "INACTIVE / SPARSE"
        else:
            conf = "Medium (Moderate Variance)"
            risk = "NORMAL HEALTHY"

        edge_case_records.append({
            "archetype_id": archetype_id,
            "archetype_name": archetype_name,
            "raw_prediction": round(raw_pred, 4),
            "calibrated_prediction": round(calib_pred, 4),
            "zero_calib_triggered": z_calib,
            "stockout_calib_triggered": stk_calib,
            "confidence": conf,
            "risk": risk,
            "expected_min": expected_bounds[0],
            "expected_max": expected_bounds[1]
        })

        # Sanity Checks
        is_nan = np.isnan(calib_pred)
        is_inf = np.isinf(calib_pred)
        is_negative = calib_pred < 0
        in_bounds = expected_bounds[0] <= calib_pred <= expected_bounds[1]

        if is_nan or is_inf or is_negative:
            recorder.record(
                test_id=test_code,
                category="Demand Edge Cases",
                test_name=f"Archetype {archetype_id}: {archetype_name} (Numerical Sanity)",
                status="FAIL",
                severity="CRITICAL",
                expected="Prediction must be finite, non-NaN, and >= 0.0",
                actual=f"Invalid prediction: {calib_pred}",
                component="model_inference",
                production_affected="YES"
            )
        elif not in_bounds:
            recorder.record(
                test_id=test_code,
                category="Demand Edge Cases",
                test_name=f"Archetype {archetype_id}: {archetype_name} (Magnitude Plausibility)",
                status="WARNING",
                severity="MEDIUM",
                expected=f"Plausible daily forecast in [{expected_bounds[0]}, {expected_bounds[1]}]",
                actual=f"Calibrated prediction: {calib_pred:.4f} units/day (Raw: {raw_pred:.4f})",
                component="model_inference",
                root_cause="Model extrapolation or calibration insufficient for extreme pattern",
                recommended_fix="Review feature momentum ratio or add domain bounding logic",
                production_affected="NO"
            )
        else:
            recorder.record(
                test_id=test_code,
                category="Demand Edge Cases",
                test_name=f"Archetype {archetype_id}: {archetype_name}",
                status="PASS",
                severity="LOW",
                expected=f"Plausible daily forecast in [{expected_bounds[0]}, {expected_bounds[1]}]",
                actual=f"Calibrated prediction: {calib_pred:.4f} units/day (Raw: {raw_pred:.4f}, Risk: {risk})",
                component="model_inference",
                production_affected="NO"
            )

    # 1. Completely zero-demand SKU
    evaluate_archetype(
        "ARCH-01", "Completely Zero-Demand SKU",
        {"v7": 0.0, "v14": 0.0, "v30": 0.0, "v60": 0.0, "v90": 0.0, "v180": 0.0, "v365": 0.0, "lag_1": 0.0, "lag_7": 0.0, "sales_days_30": 0.0, "promo_days_30": 0.0},
        (0.0, 0.15), "DEM-01"
    )

    # 2. Long zero-demand period (180 days 0, v365 had volume)
    evaluate_archetype(
        "ARCH-02", "Long Zero-Demand Period (180d zero)",
        {"v7": 0.0, "v14": 0.0, "v30": 0.0, "v60": 0.0, "v90": 0.0, "v180": 0.0, "v365": 1.5, "lag_1": 0.0, "lag_7": 0.0, "sales_days_30": 0.0},
        (0.0, 0.25), "DEM-02"
    )

    # 3. One isolated sale after long zero period
    evaluate_archetype(
        "ARCH-03", "Isolated Sale after Zero Period",
        {"lag_1": 3.0, "v7": 0.43, "v14": 0.21, "v30": 0.10, "v90": 0.03, "v180": 0.02, "sales_days_30": 1.0, "cv_30": 2.8},
        (0.0, 0.80), "DEM-03"
    )

    # 4. Sudden spike (10x normal)
    evaluate_archetype(
        "ARCH-04", "Sudden 10x Demand Spike",
        {"lag_1": 40.0, "v7": 8.0, "v14": 4.5, "v30": 2.5, "v90": 1.0, "v14_vs_v30": 1.8, "cv_30": 2.1},
        (1.0, 25.0), "DEM-04"
    )

    # 5. Multiple consecutive spikes
    evaluate_archetype(
        "ARCH-05", "Multiple Consecutive Demand Spikes",
        {"lag_1": 35.0, "lag_7": 30.0, "v7": 28.0, "v14": 20.0, "v30": 12.0, "v90": 4.0, "v14_vs_v30": 1.67, "cv_30": 1.4},
        (5.0, 45.0), "DEM-05"
    )

    # 6. Sudden demand collapse
    evaluate_archetype(
        "ARCH-06", "Sudden Demand Collapse (30 u/d -> 0)",
        {"lag_1": 0.0, "lag_7": 0.0, "v7": 1.0, "v14": 12.0, "v30": 25.0, "v90": 28.0, "v14_vs_v30": 0.48, "cv_30": 1.8},
        (0.0, 15.0), "DEM-06"
    )

    # 7. Gradual demand increase
    evaluate_archetype(
        "ARCH-07", "Gradual Demand Increase",
        {"lag_1": 15.0, "v7": 12.0, "v14": 9.0, "v30": 6.0, "v90": 3.0, "v14_vs_v30": 1.5, "cv_30": 0.6},
        (5.0, 20.0), "DEM-07"
    )

    # 8. Gradual demand decrease
    evaluate_archetype(
        "ARCH-08", "Gradual Demand Decrease",
        {"lag_1": 2.0, "v7": 3.5, "v14": 6.0, "v30": 10.0, "v90": 15.0, "v14_vs_v30": 0.60, "cv_30": 0.7},
        (1.0, 10.0), "DEM-08"
    )

    # 9. Highly volatile SKU (CV > 2.5)
    evaluate_archetype(
        "ARCH-09", "Highly Volatile SKU (CV > 2.5)",
        {"lag_1": 0.0, "v7": 3.0, "v14": 2.5, "v30": 2.8, "v90": 2.5, "cv_30": 2.7, "cv_90": 2.4},
        (0.5, 6.0), "DEM-09"
    )

    # 10. Stable high-volume SKU (CV < 0.2)
    evaluate_archetype(
        "ARCH-10", "Stable High-Volume SKU",
        {"lag_1": 52.0, "v7": 50.0, "v14": 49.5, "v30": 51.0, "v90": 48.0, "cv_30": 0.15, "cv_90": 0.18},
        (35.0, 65.0), "DEM-10"
    )

    # 11. Stable low-volume SKU
    evaluate_archetype(
        "ARCH-11", "Stable Low-Volume SKU",
        {"lag_1": 0.0, "v7": 0.28, "v14": 0.30, "v30": 0.27, "v90": 0.29, "cv_30": 0.35},
        (0.1, 0.8), "DEM-11"
    )

    # 12. Intermittent SKU
    evaluate_archetype(
        "ARCH-12", "Intermittent Burst SKU",
        {"lag_1": 0.0, "lag_7": 6.0, "v7": 1.2, "v14": 0.8, "v30": 1.1, "v90": 0.9, "cv_30": 1.6, "sales_days_30": 6.0},
        (0.2, 2.5), "DEM-12"
    )

    # 13. Weekend-only demand (is_weekend=1)
    evaluate_archetype(
        "ARCH-13", "Weekend-Only SKU (Evaluated on Saturday)",
        {"day_of_week": 5, "is_weekend": 1, "v7": 4.0, "v14": 4.1, "v30": 3.9, "cv_30": 1.5},
        (1.0, 10.0), "DEM-13"
    )

    # 14. Weekday-only demand (is_weekend=0)
    evaluate_archetype(
        "ARCH-14", "Weekday-Only SKU (Evaluated on Tuesday)",
        {"day_of_week": 1, "is_weekend": 0, "v7": 6.0, "v14": 5.8, "v30": 6.1, "cv_30": 0.8},
        (2.0, 12.0), "DEM-14"
    )

    # 15. Extremely high-volume SKU
    evaluate_archetype(
        "ARCH-15", "Extremely High-Volume SKU (500 u/d)",
        {"lag_1": 480.0, "v7": 490.0, "v14": 505.0, "v30": 500.0, "v90": 480.0, "cv_30": 0.12},
        (300.0, 700.0), "DEM-15"
    )

    # 16. Extremely low-volume SKU (0.01 u/d)
    evaluate_archetype(
        "ARCH-16", "Extremely Low-Volume SKU (0.01 u/d)",
        {"lag_1": 0.0, "v7": 0.0, "v14": 0.0, "v30": 0.03, "v90": 0.02, "v365": 0.01, "sales_days_30": 1.0},
        (0.0, 0.15), "DEM-16"
    )

    # 17. SKU with one giant transaction
    evaluate_archetype(
        "ARCH-17", "One Giant Transaction (500 u on t-7)",
        {"lag_1": 0.0, "lag_7": 500.0, "v7": 71.4, "v14": 35.7, "v30": 16.7, "v90": 5.6, "sales_days_30": 1.0, "cv_30": 3.8},
        (1.0, 50.0), "DEM-17"
    )

    # 18. Repeated small transactions (1 unit every day)
    evaluate_archetype(
        "ARCH-18", "Repeated Small Transactions (1 u/d constant)",
        {"lag_1": 1.0, "lag_7": 1.0, "v7": 1.0, "v14": 1.0, "v30": 1.0, "v90": 1.0, "cv_30": 0.01, "sales_days_30": 30.0},
        (0.6, 1.6), "DEM-18"
    )

    # Save CSV deliverable
    edge_df = pd.DataFrame(edge_case_records)
    out_csv = OUTPUTS_DIR / "edge_case_predictions.csv"
    edge_df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv} ({len(edge_df)} archetypes)")
    print(f"Suite 2 execution finished: {recorder.get_summary()}")


if __name__ == "__main__":
    rec = QARecorder("Suite 2: Demand Dynamics")
    run_demand_dynamics_tests(rec)
