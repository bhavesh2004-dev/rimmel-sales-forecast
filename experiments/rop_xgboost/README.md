# Jev + XGBoost Replenishment & Reorder Point (ROP) Prototype
## Isolated Research Experiment — Not Part of Production Pipeline

> [!IMPORTANT]
> **GOVERNANCE NOTICE**: **Exp6 production forecasting system was not modified.**  
> This prototype is strictly confined to `experiments/rop_xgboost/`. It does NOT touch the certified Exp6 LightGBM forecasting model, production features, calibration rules, database tables, validation outputs, or live dashboard deliverables.

---

## 1. Architectural Overview

This experiment evaluates Tosif's proposed replenishment architecture:

```text
Business & Platform Context (Pricing, promotions, stockouts, Amazon traffic, catalog tier)
       │
       ▼
  Jev (TypeSafe AI System 1 Model)
       │
       ▼
Structured Context Features (Urgency score, promo lift risk, stockout penalty, channel weight)
       │
       ▼
    XGBoost Quantile Regressor (Objective: reg:quantileerror, alpha=0.95)
       │
       ▼
Reorder Point (ROP) & Order-Up-To Reorder Quantity (ROQ)
```

---

## 2. Directory Structure

```text
experiments/rop_xgboost/
├── __init__.py                       # Package initializer
├── config.py                          # Lead time scenarios, service levels, and hyperparameters
├── jev_adapter.py                     # TypeSafe AI Jev adapter, input/output schemas, deterministic provider
├── rop_target.py                      # Target definition, quantile formulation, missing data audit
├── rop_feature_builder.py             # Read-only data loader and feature matrix constructor
├── baseline_rop_model.py              # Model A: Pure structured tabular XGBoost ROP
├── jev_enhanced_rop_model.py          # Model B: Structured + TypeSafe AI Jev context XGBoost ROP
├── walk_forward_backtest.py           # Day-by-day discrete-event inventory simulator (zero leakage)
├── run_experiment.py                  # Master experiment execution script
├── ROP_AUDIT_AND_EXPERIMENT_REPORT.md  # Detailed technical and business report
└── outputs/
    ├── comparison_metrics.csv         # Side-by-side performance comparison
    ├── feature_importance_jev_enhanced.csv # Feature gain rankings
    └── experiment_summary.json        # Machine-readable evaluation summary
```

---

## 3. How to Run the Experiment

To execute the complete pipeline, train both models, and run the 62-day walk-forward simulation:

```powershell
python -m experiments.rop_xgboost.run_experiment
```

---

## 4. Key Experimental Findings

1. **Jev Feature Relevance**: Jev-derived context features contributed meaningful predictive gain to XGBoost, led by `jev_stockout_severity_penalty` (Gain: 0.1198) and `jev_confidence` (Gain: 0.0378).
2. **Quantile Pinball Loss**: Jev-enhanced XGBoost achieved a **0.91% reduction in Pinball Loss** (0.5104 vs. 0.5151), demonstrating tighter quantile estimation.
3. **Replenishment Behavior**: The Jev-enhanced model proactively triggered 16 additional purchase orders for high-revenue hero products, elevating the overall unit service level to **98.67%** while maintaining a low **0.91% stockout rate**.
4. **Supply Chain Data Gaps**: Supplier lead times ($L$), minimum order quantities (MOQ), and unit holding costs are not recorded in the client's sales files, requiring formal business alignment before production consideration.
