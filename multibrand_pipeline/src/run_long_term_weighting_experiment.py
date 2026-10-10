"""
Long-Term Demand Feature Weighting Experiment Runner
Tightly controlled experiment comparing:
- Model A: Baseline (feature_contri = 1.0 for all)
- Model B: Moderate Long-Term Emphasis (Short: 0.95, Medium: 1.10, Long: 1.20, Other: 1.00)
- Model C: Stronger Long-Term Emphasis (Short: 0.90, Medium: 1.15, Long: 1.30, Other: 1.00)

Evaluated across:
- 3 Walk-forward Development Windows (June 1-10, June 21-July 1, July 10-20, 2026)
- Locked Holdout September Benchmark (September 1-10, 2026)
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("weighting_experiment")

def calc_metrics(act, prd):
    act = np.asarray(act, dtype=float)
    prd = np.asarray(prd, dtype=float)
    tot_act = float(np.sum(act))
    tot_prd = float(np.sum(prd))
    abs_err = float(np.sum(np.abs(act - prd)))
    wape = (abs_err / tot_act * 100.0) if tot_act > 0 else 0.0
    bias = ((tot_prd - tot_act) / tot_act * 100.0) if tot_act > 0 else 0.0
    mae = float(np.mean(np.abs(act - prd)))
    rmse = float(np.sqrt(np.mean((act - prd) ** 2)))
    return {
        "actual": round(tot_act, 2),
        "predicted": round(tot_prd, 2),
        "abs_error": round(abs_err, 2),
        "wape": round(wape, 2),
        "bias": round(bias, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4)
    }

def main():
    config_path = "multibrand_pipeline/config/pipeline_config.yaml"
    schema_path = "multibrand_pipeline/config/feature_schema.json"
    parquet_path = "experiments/global_multibrand_lgbm/data/global_training_dataset.parquet"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(schema_path, "r") as f:
        schema = json.load(f)

    feats = schema['feature_list']
    cats = schema['categorical_features']
    model_params = config['model_parameters'].copy()
    calib_cfg = config.get('calibration', {})
    alpha = calib_cfg.get('zero_demand_alpha', 0.10)
    beta = calib_cfg.get('stockout_beta', 0.10)

    # Define Feature Groups
    group_a = ['lag_1', 'lag_7', 'lag_14', 'v7', 'v14']
    group_b = ['lag_30', 'v30', 'v60', 'sales_days_30', 'sales_days_90', 'v14_vs_v30', 'v30_vs_v90', 'v30_vs_v180']
    group_c = [
        'lag_90', 'lag_180', 'lag_365', 'v90', 'v180', 'v365',
        'v30_vs_v365', 'v90_vs_v365', 'same_period_last_year_7d',
        'same_period_last_year_30d', 'yoy_7d', 'yoy_30d'
    ]

    other_feats = [f for f in feats if f not in group_a and f not in group_b and f not in group_c]

    # Build Weight Vectors
    weights_model_a = [1.0] * len(feats)

    weights_model_b = []
    for f in feats:
        if f in group_a:
            weights_model_b.append(0.95)
        elif f in group_b:
            weights_model_b.append(1.10)
        elif f in group_c:
            weights_model_b.append(1.20)
        else:
            weights_model_b.append(1.00)

    weights_model_c = []
    for f in feats:
        if f in group_a:
            weights_model_c.append(0.90)
        elif f in group_b:
            weights_model_c.append(1.15)
        elif f in group_c:
            weights_model_c.append(1.30)
        else:
            weights_model_c.append(1.00)

    configs = {
        "Model_A_Baseline": weights_model_a,
        "Model_B_Moderate": weights_model_b,
        "Model_C_Stronger": weights_model_c
    }

    logger.info("Loading parquet dataset...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows.")

    dev_windows = [
        {"name": "Window_1", "train_start": "2025-08-01", "train_end": "2026-05-31", "eval_start": "2026-06-01", "eval_end": "2026-06-10"},
        {"name": "Window_2", "train_start": "2025-08-01", "train_end": "2026-06-20", "eval_start": "2026-06-21", "eval_end": "2026-07-01"},
        {"name": "Window_3", "train_start": "2025-08-01", "train_end": "2026-07-09", "eval_start": "2026-07-10", "eval_end": "2026-07-20"},
    ]

    benchmark_window = {
        "name": "September_Benchmark",
        "train_start": "2025-08-01",
        "train_end": "2026-08-31",
        "eval_start": "2026-09-01",
        "eval_end": "2026-09-10"
    }

    all_windows = dev_windows + [benchmark_window]
    experiment_results = {
        "groups": {
            "group_a_short": group_a,
            "group_b_medium": group_b,
            "group_c_long": group_c,
            "other_features": other_feats
        },
        "windows": {}
    }

    for win in all_windows:
        win_name = win["name"]
        t_start, t_end = win["train_start"], win["train_end"]
        e_start, e_end = win["eval_start"], win["eval_end"]

        logger.info(f"\n==================================================")
        logger.info(f"PROCESSING {win_name}: Train {t_start}->{t_end} | Eval {e_start}->{e_end}")
        logger.info(f"==================================================")

        train_mask = (df['date'] >= t_start) & (df['date'] <= t_end)
        eval_mask = (df['date'] >= e_start) & (df['date'] <= e_end)

        tr_sub = df[train_mask].copy()
        ev_sub = df[eval_mask].copy()

        logger.info(f"Train rows: {len(tr_sub):,}, Eval rows: {len(ev_sub):,}")

        X_tr = tr_sub[feats].copy()
        for c in cats:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype('category')
        y_tr = tr_sub['model_units_sold'].values

        X_ev = ev_sub[feats].copy()
        for c in cats:
            if c in X_ev.columns:
                X_ev[c] = X_ev[c].astype('category')
        y_ev = ev_sub['model_units_sold'].values

        z_mask = (ev_sub['v7'] == 0) & (ev_sub['v14'] == 0) & (ev_sub['v30'] == 0)
        stk_mask = (ev_sub['in_stock_flag'] == 0)

        win_dict = {}

        for cfg_name, weight_vec in configs.items():
            logger.info(f"Training {cfg_name} on {win_name}...")
            # If all weights 1.0, don't need to pass feature_contri, but passing it is identical
            reg = lgb.LGBMRegressor(**model_params, feature_contri=weight_vec)
            reg.fit(X_tr, y_tr)

            raw_preds = np.clip(reg.predict(X_ev), 0, None)

            # Exp6 Calibration
            calib_preds = raw_preds.copy()
            calib_preds[z_mask.values] *= alpha
            calib_preds[stk_mask.values] *= beta

            # Combined metrics
            comb_raw = calc_metrics(y_ev, raw_preds)
            comb_cal = calc_metrics(y_ev, calib_preds)

            # Brand breakdowns
            brand_metrics_raw = {}
            brand_metrics_cal = {}
            for b_id in ["RIMMEL", "MAX_FACTOR"]:
                b_mask = (ev_sub['brand_id'] == b_id).values
                brand_metrics_raw[b_id] = calc_metrics(y_ev[b_mask], raw_preds[b_mask])
                brand_metrics_cal[b_id] = calc_metrics(y_ev[b_mask], calib_preds[b_mask])

            # Platform breakdowns
            plat_metrics_raw = {}
            plat_metrics_cal = {}
            for p_grp in ["Amazon", "eBay", "Website", "Other"]:
                p_mask = (ev_sub['platform_group'] == p_grp).values
                plat_metrics_raw[p_grp] = calc_metrics(y_ev[p_mask], raw_preds[p_mask])
                plat_metrics_cal[p_grp] = calc_metrics(y_ev[p_mask], calib_preds[p_mask])

            # Diagnostics
            act_zero = float((y_ev == 0).mean() * 100.0)
            pred_zero_raw = float((raw_preds == 0).mean() * 100.0)
            pred_zero_cal = float((calib_preds == 0).mean() * 100.0)
            fp_raw = int(((y_ev == 0) & (raw_preds >= 1.0)).sum())
            fp_cal = int(((y_ev == 0) & (calib_preds >= 1.0)).sum())
            act_under = int(((y_ev > 0) & (calib_preds < y_ev)).sum())
            act_over = int(((y_ev > 0) & (calib_preds > y_ev)).sum())

            # Feature gain analysis
            splits = reg.booster_.feature_importance(importance_type='split')
            gains = reg.booster_.feature_importance(importance_type='gain')
            tot_gain = float(np.sum(gains)) if np.sum(gains) > 0 else 1.0

            feat_details = []
            gain_by_group = {"group_a_short": 0.0, "group_b_medium": 0.0, "group_c_long": 0.0, "other": 0.0}
            splits_by_group = {"group_a_short": 0, "group_b_medium": 0, "group_c_long": 0, "other": 0}

            for idx, col in enumerate(feats):
                sp = int(splits[idx])
                gn = float(gains[idx])
                gn_share = round((gn / tot_gain) * 100.0, 2)
                feat_details.append({"feature": col, "split": sp, "gain": gn, "gain_share": gn_share})

                if col in group_a:
                    gain_by_group["group_a_short"] += gn
                    splits_by_group["group_a_short"] += sp
                elif col in group_b:
                    gain_by_group["group_b_medium"] += gn
                    splits_by_group["group_b_medium"] += sp
                elif col in group_c:
                    gain_by_group["group_c_long"] += gn
                    splits_by_group["group_c_long"] += sp
                else:
                    gain_by_group["other"] += gn
                    splits_by_group["other"] += sp

            gain_share_by_group = {
                k: round((v / tot_gain) * 100.0, 2) for k, v in gain_by_group.items()
            }

            # Top 10 features by gain
            top10 = sorted(feat_details, key=lambda x: x["gain"], reverse=True)[:10]

            win_dict[cfg_name] = {
                "combined_raw": comb_raw,
                "combined_calib": comb_cal,
                "by_brand_raw": brand_metrics_raw,
                "by_brand_calib": brand_metrics_cal,
                "by_platform_raw": plat_metrics_raw,
                "by_platform_calib": plat_metrics_cal,
                "diagnostics": {
                    "act_zero_rate": round(act_zero, 2),
                    "pred_zero_raw": round(pred_zero_raw, 2),
                    "pred_zero_cal": round(pred_zero_cal, 2),
                    "fp_rows_raw": fp_raw,
                    "fp_rows_cal": fp_cal,
                    "active_under": act_under,
                    "active_over": act_over
                },
                "group_gain_summary": {
                    "total_gain": round(tot_gain, 2),
                    "gain_by_group": {k: round(v, 2) for k, v in gain_by_group.items()},
                    "gain_share_by_group": gain_share_by_group,
                    "splits_by_group": splits_by_group
                },
                "top_10_features": top10
            }

        experiment_results["windows"][win_name] = win_dict

    # Compute Development Averages across Window 1, Window 2, Window 3
    dev_names = ["Window_1", "Window_2", "Window_3"]
    dev_avg = {}
    for cfg in configs.keys():
        avg_wape_raw = float(np.mean([experiment_results["windows"][w][cfg]["combined_raw"]["wape"] for w in dev_names]))
        avg_wape_cal = float(np.mean([experiment_results["windows"][w][cfg]["combined_calib"]["wape"] for w in dev_names]))
        avg_bias_raw = float(np.mean([experiment_results["windows"][w][cfg]["combined_raw"]["bias"] for w in dev_names]))
        avg_bias_cal = float(np.mean([experiment_results["windows"][w][cfg]["combined_calib"]["bias"] for w in dev_names]))
        avg_rim_cal_wape = float(np.mean([experiment_results["windows"][w][cfg]["by_brand_calib"]["RIMMEL"]["wape"] for w in dev_names]))
        avg_mf_cal_wape = float(np.mean([experiment_results["windows"][w][cfg]["by_brand_calib"]["MAX_FACTOR"]["wape"] for w in dev_names]))
        avg_mae = float(np.mean([experiment_results["windows"][w][cfg]["combined_calib"]["mae"] for w in dev_names]))
        avg_rmse = float(np.mean([experiment_results["windows"][w][cfg]["combined_calib"]["rmse"] for w in dev_names]))

        dev_avg[cfg] = {
            "avg_wape_raw": round(avg_wape_raw, 2),
            "avg_wape_calib": round(avg_wape_cal, 2),
            "avg_bias_raw": round(avg_bias_raw, 2),
            "avg_bias_calib": round(avg_bias_cal, 2),
            "avg_rimmel_calib_wape": round(avg_rim_cal_wape, 2),
            "avg_max_factor_calib_wape": round(avg_mf_cal_wape, 2),
            "avg_mae_calib": round(avg_mae, 4),
            "avg_rmse_calib": round(avg_rmse, 4)
        }
    experiment_results["development_averages"] = dev_avg

    out_json = "multibrand_pipeline/reports/weighting_experiment_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(f"Experiment complete! Results saved to {out_json}")
    print("\n--- DEVELOPMENT AVERAGES ---")
    for k, v in dev_avg.items():
        print(f"{k}: WAPE Calib = {v['avg_wape_calib']}%, Bias Calib = {v['avg_bias_calib']}%, Rimmel WAPE = {v['avg_rimmel_calib_wape']}%, MF WAPE = {v['avg_max_factor_calib_wape']}%")

    print("\n--- SEPTEMBER BENCHMARK ---")
    sep_res = experiment_results["windows"]["September_Benchmark"]
    for k in configs.keys():
        s = sep_res[k]["combined_calib"]
        r = sep_res[k]["by_brand_calib"]["RIMMEL"]
        m = sep_res[k]["by_brand_calib"]["MAX_FACTOR"]
        print(f"{k}: Sep WAPE Calib = {s['wape']}%, Bias = {s['bias']}%, Rimmel = {r['wape']}%, MF = {m['wape']}%")

if __name__ == "__main__":
    main()
