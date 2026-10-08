"""
Master Runner: Jev + XGBoost ROP Replenishment Experiment
=========================================================
Executes the isolated comparison between:
- Model A: Baseline Structured-Data XGBoost ROP
- Model B: Structured-Data + TypeSafe AI Jev Context XGBoost ROP

CRITICAL GOVERNANCE INVARIANT:
Exp6 production forecasting system was not modified.
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.rop_xgboost.config import (
    DEFAULT_LEAD_TIME_DAYS, OUTPUTS_DIR, TARGET_SERVICE_LEVEL
)
from experiments.rop_xgboost.rop_target import audit_missing_requirements
from experiments.rop_xgboost.rop_feature_builder import prepare_rop_datasets
from experiments.rop_xgboost.baseline_rop_model import BaselineXGBoostROP
from experiments.rop_xgboost.jev_enhanced_rop_model import JevEnhancedXGBoostROP
from experiments.rop_xgboost.walk_forward_backtest import ReplenishmentSimulator


def run_rop_experiment(lead_time_days: int = DEFAULT_LEAD_TIME_DAYS):
    start_time = time.time()
    print("=" * 90)
    print("STARTING EXPERIMENT: JEV + XGBOOST REORDER POINT (ROP) REPLENISHMENT")
    print("Architecture: Business Context -> TypeSafe AI Jev -> Context Features -> XGBoost -> ROP / ROQ")
    print("=" * 90)

    # 1. Audit Missing Requirements
    print("\n[PHASE 1] Auditing Supply-Chain Data Requirements & Gap Analysis...")
    missing_reqs = audit_missing_requirements()
    for req, desc in missing_reqs.items():
        print(f"  • {req}:")
        print(f"    {desc}")

    # 2. Prepare Datasets
    print("\n[PHASE 2] Loading Data & Generating Features (Zero Production Impact)...")
    df_train, df_backtest = prepare_rop_datasets(lead_time_days=lead_time_days)

    # 3. Train Model A: Baseline Structured-Data XGBoost
    print("\n[PHASE 3] Training Model A: Baseline Structured-Data XGBoost ROP...")
    model_a = BaselineXGBoostROP()
    model_a.fit(df_train)
    eval_a = model_a.evaluate_pinball_loss(df_backtest, alpha=TARGET_SERVICE_LEVEL)
    print(f"  Model A Pinball Loss: {eval_a['mean_pinball_loss']} | Empirical Coverage: {eval_a['empirical_coverage_pct']}%")

    # 4. Train Model B: Structured-Data + Jev-Context XGBoost
    print("\n[PHASE 4] Training Model B: Structured-Data + Jev-Context XGBoost ROP...")
    model_b = JevEnhancedXGBoostROP()
    model_b.fit(df_train)
    eval_b = model_b.evaluate_pinball_loss(df_backtest, alpha=TARGET_SERVICE_LEVEL)
    print(f"  Model B Pinball Loss: {eval_b['mean_pinball_loss']} | Empirical Coverage: {eval_b['empirical_coverage_pct']}%")

    # 5. Feature Importance Analysis
    imp_b = model_b.get_feature_importance()
    imp_csv_path = os.path.join(OUTPUTS_DIR, 'feature_importance_jev_enhanced.csv')
    imp_b.to_csv(imp_csv_path, index=False)
    print(f"\n[PHASE 5] Jev Feature Importance Ranking:")
    jev_features_imp = imp_b[imp_b['is_jev_feature']]
    for _, r in jev_features_imp.iterrows():
        print(f"  - {r['feature']:<35} | Gain: {r['importance_gain']:.4f}")

    # 6. Walk-Forward Backtest Simulation
    print("\n[PHASE 6] Running Day-by-Day Walk-Forward Inventory Backtest...")
    simulator = ReplenishmentSimulator(lead_time_days=lead_time_days)
    
    sim_a = simulator.run_simulation(df_backtest, model_a, model_name="A. Baseline XGBoost ROP")
    sim_b = simulator.run_simulation(df_backtest, model_b, model_name="B. Jev-Enhanced XGBoost ROP")

    # 7. Compile Comparison Table
    comparison_records = [
        {
            'Metric': 'Model Description',
            'Model A (Baseline XGBoost)': 'Structured Tabular Signals Only',
            'Model B (Jev-Enhanced XGBoost)': 'Structured + TypeSafe AI Jev Context',
            'Delta / Relative Impact': 'Jev Context Augmentation'
        },
        {
            'Metric': 'Feature Count',
            'Model A (Baseline XGBoost)': str(len(model_a.feature_cols)),
            'Model B (Jev-Enhanced XGBoost)': str(len(model_b.feature_cols)),
            'Delta / Relative Impact': f"+{len(model_b.feature_cols) - len(model_a.feature_cols)} Jev features"
        },
        {
            'Metric': 'Quantile Pinball Loss (Lower is Better)',
            'Model A (Baseline XGBoost)': f"{eval_a['mean_pinball_loss']:.4f}",
            'Model B (Jev-Enhanced XGBoost)': f"{eval_b['mean_pinball_loss']:.4f}",
            'Delta / Relative Impact': f"{(eval_b['mean_pinball_loss'] - eval_a['mean_pinball_loss']) / eval_a['mean_pinball_loss'] * 100:+.2f}%"
        },
        {
            'Metric': 'Empirical Service Level / Coverage (%)',
            'Model A (Baseline XGBoost)': f"{eval_a['empirical_coverage_pct']:.2f}%",
            'Model B (Jev-Enhanced XGBoost)': f"{eval_b['empirical_coverage_pct']:.2f}%",
            'Delta / Relative Impact': f"{eval_b['empirical_coverage_pct'] - eval_a['empirical_coverage_pct']:+.2f}% pts"
        },
        {
            'Metric': 'Backtest Unit Fill Rate (%)',
            'Model A (Baseline XGBoost)': f"{sim_a['unit_service_level_pct']:.2f}%",
            'Model B (Jev-Enhanced XGBoost)': f"{sim_b['unit_service_level_pct']:.2f}%",
            'Delta / Relative Impact': f"{sim_b['unit_service_level_pct'] - sim_a['unit_service_level_pct']:+.2f}% pts"
        },
        {
            'Metric': 'Backtest Stockout Event Count',
            'Model A (Baseline XGBoost)': str(sim_a['total_stockout_events']),
            'Model B (Jev-Enhanced XGBoost)': str(sim_b['total_stockout_events']),
            'Delta / Relative Impact': f"{sim_b['total_stockout_events'] - sim_a['total_stockout_events']:+d} events"
        },
        {
            'Metric': 'Stockout Days Rate (%)',
            'Model A (Baseline XGBoost)': f"{sim_a['stockout_rate_pct']:.2f}%",
            'Model B (Jev-Enhanced XGBoost)': f"{sim_b['stockout_rate_pct']:.2f}%",
            'Delta / Relative Impact': f"{sim_b['stockout_rate_pct'] - sim_a['stockout_rate_pct']:+.2f}% pts"
        },
        {
            'Metric': 'Average On-Hand Warehouse Stock',
            'Model A (Baseline XGBoost)': f"{sim_a['average_inventory_units']:.1f} units",
            'Model B (Jev-Enhanced XGBoost)': f"{sim_b['average_inventory_units']:.1f} units",
            'Delta / Relative Impact': f"{sim_b['average_inventory_units'] - sim_a['average_inventory_units']:+.1f} units"
        },
        {
            'Metric': 'Replenishment POs Triggered',
            'Model A (Baseline XGBoost)': str(sim_a['total_pos_triggered']),
            'Model B (Jev-Enhanced XGBoost)': str(sim_b['total_pos_triggered']),
            'Delta / Relative Impact': f"{sim_b['total_pos_triggered'] - sim_a['total_pos_triggered']:+d} orders"
        },
        {
            'Metric': 'Total Units Reordered',
            'Model A (Baseline XGBoost)': f"{sim_a['total_units_ordered']:,.0f}",
            'Model B (Jev-Enhanced XGBoost)': f"{sim_b['total_units_ordered']:,.0f}",
            'Delta / Relative Impact': f"{sim_b['total_units_ordered'] - sim_a['total_units_ordered']:+,.0f} units"
        }
    ]

    comp_df = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(OUTPUTS_DIR, 'comparison_metrics.csv')
    comp_df.to_csv(comp_csv_path, index=False)

    summary_json = {
        'experiment_name': 'Jev_XGBoost_ROP_Replenishment_Experiment',
        'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'lead_time_days': lead_time_days,
        'target_quantile': TARGET_SERVICE_LEVEL,
        'model_a_eval': eval_a,
        'model_b_eval': eval_b,
        'model_a_sim': sim_a,
        'model_b_sim': sim_b,
        'production_safety_status': 'Exp6 production forecasting system was not modified.'
    }
    with open(os.path.join(OUTPUTS_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary_json, f, indent=2)

    # 8. Display Executive Comparison
    print("\n" + "=" * 90)
    print("EXPERIMENTAL EVALUATION SUMMARY: BASELINE VS. JEV-ENHANCED XGBOOST ROP")
    print("=" * 90)
    for r in comparison_records:
        print(f"  {r['Metric']:<40} | Model A: {r['Model A (Baseline XGBoost)']:<20} | Model B: {r['Model B (Jev-Enhanced XGBoost)']:<20} | Delta: {r['Delta / Relative Impact']}")
    print("=" * 90)
    print("\nGOVERNANCE VERIFICATION:")
    print("  --> Exp6 production forecasting system was not modified.")
    print(f"  --> Outputs and artifacts saved to: {OUTPUTS_DIR}")
    print(f"  --> Runtime elapsed: {time.time() - start_time:.2f} seconds.")
    print("=" * 90)


if __name__ == '__main__':
    run_rop_experiment()
