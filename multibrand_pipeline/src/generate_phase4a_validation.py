"""
PHASE 4A — Locked September 1–10 Validation Runner & Report Generator
Executes:
1. Training strictly on 2025-08-01 to 2026-08-31 (Sep 1-10 completely unseen)
2. Day-by-day forward simulation for 2026-09-01 to 2026-09-10
3. Aggregation of actual physical sales units across platforms for Sep 1-10
4. Creation of SEP01_10_VALIDATION_FORECAST_COMPARISON.xlsx (14 columns, 1 sheet)
5. Comprehensive metrics calculation and forensic reconciliation against Phase 3
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Ensure pipeline root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(PIPELINE_ROOT)

if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.train import prepare_feature_matrix
from src.inventory import classify_inventory_risk_and_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase4a_validation")

# Openpyxl Styles
FONT_TITLE = Font(name='Segoe UI', size=13, bold=True, color='FFFFFF')
FONT_SUBTITLE = Font(name='Segoe UI', size=9, italic=True, color='E0E0E0')
FONT_HEADER = Font(name='Segoe UI', size=9, bold=True, color='FFFFFF')
FONT_BODY = Font(name='Segoe UI', size=9, color='000000')

FILL_NAVY = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
FILL_STEEL = PatternFill(start_color='2F5597', end_color='2F5597', fill_type='solid')
FILL_HEADER = PatternFill(start_color='333F48', end_color='333F48', fill_type='solid')
FILL_TOT_HDR = PatternFill(start_color='1E3A8A', end_color='1E3A8A', fill_type='solid')
FILL_ACT_HDR = PatternFill(start_color='107C41', end_color='107C41', fill_type='solid')
FILL_ZEBRA = PatternFill(start_color='F9FAFB', end_color='F9FAFB', fill_type='solid')
FILL_WHITE = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')

BORDER_THIN = Border(
    left=Side(style='thin', color='E5E7EB'),
    right=Side(style='thin', color='E5E7EB'),
    top=Side(style='thin', color='E5E7EB'),
    bottom=Side(style='thin', color='E5E7EB')
)

def compute_metrics(act, prd):
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
    config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
    schema_path = os.path.join(PIPELINE_ROOT, "config", "feature_schema.json")
    parquet_path = os.path.join(WORKSPACE_ROOT, "experiments", "global_multibrand_lgbm", "data", "global_training_dataset.parquet")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feats = schema['feature_list']
    cats = schema['categorical_features']
    model_params = config['model_parameters'].copy()
    alpha = config['calibration']['zero_demand_alpha']
    beta = config['calibration']['stockout_beta']

    logger.info("Loading parquet dataset...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} total rows.")

    # 1. Dataset Partitioning
    train_mask = (df['date'] >= "2025-08-01") & (df['date'] <= "2026-08-31")
    bench_mask = (df['date'] >= "2026-09-01") & (df['date'] <= "2026-09-10")

    tr_df = df[train_mask].copy()
    be_df = df[bench_mask].copy()

    logger.info(f"Training Period: 2025-08-01 to 2026-08-31 ({len(tr_df):,} rows)")
    logger.info(f"Validation Period: 2026-09-01 to 2026-09-10 ({len(be_df):,} rows)")
    assert len(tr_df) == 891396, f"Expected 891,396 training rows, got {len(tr_df)}"
    assert len(be_df) == 22510, f"Expected 22,510 benchmark rows, got {len(be_df)}"

    # Check temporal separation
    assert tr_df['date'].max() == "2026-08-31", "Training data leaked past August 31!"
    assert be_df['date'].min() == "2026-09-01", "Benchmark data started before September 1!"

    # 2. Train Model on Training Period Only
    logger.info("Fitting Global LightGBM on 2025-08-01 to 2026-08-31...")
    X_tr = prepare_feature_matrix(tr_df, feats, cats)
    y_tr = tr_df['model_units_sold'].values
    model = lgb.LGBMRegressor(**model_params).fit(X_tr, y_tr)
    logger.info("Model fitting complete.")

    # 3. Ground Truth Actual Units (Sep 1 to Sep 10)
    actual_sku = be_df.groupby(['brand_id', 'canonical_sku'])['model_units_sold'].sum().reset_index()
    actual_sku.rename(columns={'model_units_sold': 'Actual Units'}, inplace=True)
    actual_sku['Actual Units'] = np.round(actual_sku['Actual Units']).astype(int)

    logger.info(f"Ground truth actuals computed for {len(actual_sku):,} SKUs.")
    tot_actual = actual_sku['Actual Units'].sum()
    logger.info(f"Total Actual Units across all platforms: {tot_actual:,}")
    assert tot_actual == 2627, f"Expected 2,627 actual units, got {tot_actual}"

    # 4. Forward Simulation (Sep 1 to Sep 10, from Anchor Aug 31)
    anchor_df = df[df['date'] == "2026-08-31"].copy()
    logger.info(f"Anchor catalog state contains {len(anchor_df):,} platform series on 2026-08-31.")

    fwd_dates = pd.date_range("2026-09-01", "2026-09-10", freq='D')
    sim_records = []
    for f_dt in fwd_dates:
        f_str = f_dt.strftime('%Y-%m-%d')
        dow = f_dt.dayofweek
        sim = anchor_df.copy()
        sim['date'] = f_str
        sim['day_of_week'] = dow
        X_sim = prepare_feature_matrix(sim, feats, cats)
        raw_pred = np.clip(model.predict(X_sim), 0, None)

        calib_pred = raw_pred.copy()
        z_mask = (sim['v7'] == 0) & (sim['v14'] == 0) & (sim['v30'] == 0)
        stk_mask = (sim['in_stock_flag'] == 0)
        calib_pred[z_mask.values] *= alpha
        calib_pred[stk_mask.values] *= beta

        sim['calib_daily_prediction'] = calib_pred
        sim_records.append(sim)

    df_daily_sim = pd.concat(sim_records, ignore_index=True)
    logger.info(f"Generated {len(df_daily_sim):,} forward simulated daily rows.")

    # Aggregate to 10-day SKU x Platform level
    if 'product_title' not in df_daily_sim.columns:
        df_daily_sim['product_title'] = df_daily_sim.get('category', df_daily_sim['canonical_sku'])
    if 'category' not in df_daily_sim.columns:
        df_daily_sim['category'] = df_daily_sim.get('product_title', df_daily_sim['canonical_sku'])

    series_agg = df_daily_sim.groupby(['brand_id', 'canonical_sku', 'product_title', 'category', 'platform_group']).agg(
        expected_demand_10d=('calib_daily_prediction', 'sum'),
        current_stock=('current_stock', 'first'),
        in_stock_flag=('in_stock_flag', 'first'),
        v7=('v7', 'first'),
        v14=('v14', 'first'),
        v30=('v30', 'first'),
        v90=('v90', 'first'),
        cv_30=('cv_30', 'first')
    ).reset_index()

    series_agg['recommended_10d_units'] = np.round(series_agg['expected_demand_10d']).astype(int)

    # Pivot platforms into Amazon, eBay, Website, Other
    plat_pivot = series_agg.pivot_table(
        index=['brand_id', 'canonical_sku'],
        columns='platform_group',
        values='recommended_10d_units',
        fill_value=0
    ).reset_index()

    for p in ['Amazon', 'eBay', 'Website', 'Other']:
        if p not in plat_pivot.columns:
            plat_pivot[p] = 0

    plat_pivot.rename(columns={
        'Amazon': 'Amazon Predicted',
        'eBay': 'eBay Predicted',
        'Website': 'Website Predicted',
        'Other': 'Other Predicted'
    }, inplace=True)

    plat_pivot['10-Day Forecast'] = (
        plat_pivot['Amazon Predicted'] + plat_pivot['eBay Predicted'] +
        plat_pivot['Website Predicted'] + plat_pivot['Other Predicted']
    )

    # Attach SKU metadata
    sku_meta = series_agg.groupby(['brand_id', 'canonical_sku']).agg(
        product_title=('product_title', 'first'),
        category=('category', 'first'),
        current_stock=('current_stock', 'first'),
        in_stock_flag=('in_stock_flag', 'first'),
        v7=('v7', 'first'),
        v14=('v14', 'first'),
        v30=('v30', 'first'),
        v90=('v90', 'first'),
        cv_30=('cv_30', lambda s: float(s.fillna(0.0).iloc[0]))
    ).reset_index()

    fwd_sku = pd.merge(sku_meta, plat_pivot, on=['brand_id', 'canonical_sku'], how='left')
    fwd_sku['Forecast Period'] = "2026-09-01 to 2026-09-10"

    # Merge with Ground Truth Actual Units
    val_table = pd.merge(fwd_sku, actual_sku, on=['brand_id', 'canonical_sku'], how='left')
    val_table['Actual Units'] = val_table['Actual Units'].fillna(0).astype(int)

    # Compute Confidence, Risk, Recommended Action, Reason
    records = []
    for _, r in val_table.iterrows():
        stock = float(r.get('current_stock', 0.0))
        in_stock = int(r.get('in_stock_flag', 1 if stock > 0 else 0))
        cv = float(r.get('cv_30', 0.0))
        fwd = float(r.get('10-Day Forecast', 0.0))
        v30 = float(r.get('v30', 0.0))
        v90 = float(r.get('v90', 0.0))

        risk, action = classify_inventory_risk_and_action(stock, in_stock, fwd, v90, cv, config)

        if in_stock == 0 or stock <= 0:
            conf = 'LOW'
            reason = 'Inventory availability is limiting observed demand.'
        elif cv > 1.2:
            conf = 'LOW'
            reason = 'Demand has been volatile historically, reducing forecast certainty.'
        elif fwd >= 5.0 and cv <= 0.6:
            conf = 'HIGH'
            reason = 'Stable historical demand with recent momentum remaining consistent.'
        elif fwd < 1.0 and v30 < 0.2:
            conf = 'HIGH'
            reason = 'Consistently low historical sales volume.'
        else:
            conf = 'MEDIUM'
            reason = 'Standard baseline demand run rate.'

        brand = r['brand_id']
        sku = r['canonical_sku']
        prod = r['category'] if pd.notna(r['category']) and r['category'] != '' else r['product_title']

        records.append({
            'Brand': brand,
            'SKU': sku,
            'Product': prod,
            'Forecast Period': r['Forecast Period'],
            'Amazon Predicted': int(r['Amazon Predicted']),
            'eBay Predicted': int(r['eBay Predicted']),
            'Website Predicted': int(r['Website Predicted']),
            'Other Predicted': int(r['Other Predicted']),
            '10-Day Forecast': int(r['10-Day Forecast']),
            'Actual Units': int(r['Actual Units']),
            'Confidence': conf,
            'Risk': risk,
            'Recommended Action': action,
            'Reason': reason
        })

    df_out = pd.DataFrame(records)

    # Expected 14 columns
    cols_14 = [
        'Brand', 'SKU', 'Product', 'Forecast Period',
        'Amazon Predicted', 'eBay Predicted', 'Website Predicted', 'Other Predicted',
        '10-Day Forecast', 'Actual Units', 'Confidence', 'Risk', 'Recommended Action', 'Reason'
    ]
    df_out = df_out[cols_14]
    logger.info(f"Validation table generated for {len(df_out):,} SKUs across 14 columns.")
    assert len(df_out) == 1108, f"Expected 1,108 SKUs, got {len(df_out)}"

    # 5. Export to Excel (SEP01_10_VALIDATION_FORECAST_COMPARISON.xlsx)
    excel_path = os.path.join(PIPELINE_ROOT, "reports", "SEP01_10_VALIDATION_FORECAST_COMPARISON.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    logger.info(f"Exporting validation Excel workbook to: {excel_path}...")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Validation"
    ws.views.sheetView[0].showGridLines = True

    # Title & Subtitle
    last_col = get_column_letter(len(cols_14))
    ws.merge_cells(f'A1:{last_col}1')
    ws['A1'] = "MULTI-BRAND DEMAND FORECASTING — SEPTEMBER 1–10, 2026 VALIDATION COMPARISON"
    ws['A1'].font = FONT_TITLE
    ws['A1'].fill = FILL_NAVY
    ws['A1'].alignment = Alignment(horizontal='left', vertical='center', indent=1)
    ws.row_dimensions[1].height = 28

    ws.merge_cells(f'A2:{last_col}2')
    ws['A2'] = "Model Trained: 2025-08-01 to 2026-08-31 | Unseen Validation: 2026-09-01 to 2026-09-10 | Forecast vs Actual Comparison"
    ws['A2'].font = FONT_SUBTITLE
    ws['A2'].fill = FILL_STEEL
    ws['A2'].alignment = Alignment(horizontal='left', vertical='center', indent=1)
    ws.row_dimensions[2].height = 20

    # Headers
    ws.row_dimensions[3].height = 24
    for c_idx, h in enumerate(cols_14, 1):
        cell = ws.cell(3, c_idx, h)
        cell.font = FONT_HEADER
        if h == '10-Day Forecast':
            cell.fill = FILL_TOT_HDR
        elif h == 'Actual Units':
            cell.fill = FILL_ACT_HDR
        else:
            cell.fill = FILL_HEADER
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = BORDER_THIN

    # Data Rows
    row_num = 4
    for _, r in df_out.iterrows():
        fill = FILL_ZEBRA if row_num % 2 == 0 else FILL_WHITE
        ws.row_dimensions[row_num].height = 19
        for c_idx, h in enumerate(cols_14, 1):
            val = r[h]
            cell = ws.cell(row_num, c_idx, val)
            cell.font = FONT_BODY
            cell.fill = fill
            cell.border = BORDER_THIN

            if isinstance(val, (int, float)):
                cell.number_format = '#,##0'
                cell.alignment = Alignment(horizontal='right', vertical='center')
            else:
                cell.alignment = Alignment(horizontal='left', vertical='center')
        row_num += 1

    # Column Widths
    for c_idx in range(1, len(cols_14) + 1):
        col_letter = get_column_letter(c_idx)
        max_len = max([len(str(ws.cell(r, c_idx).value or '')) for r in range(3, min(row_num, 50))] or [12])
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 45)

    wb.save(excel_path)
    logger.info(f"Saved validation workbook to: {excel_path}")

    # 6. Compute Validation Metrics
    comb_metrics = compute_metrics(df_out['Actual Units'].values, df_out['10-Day Forecast'].values)
    rim_sub = df_out[df_out['Brand'] == 'RIMMEL']
    mf_sub = df_out[df_out['Brand'] == 'MAX_FACTOR']

    rim_metrics = compute_metrics(rim_sub['Actual Units'].values, rim_sub['10-Day Forecast'].values)
    mf_metrics = compute_metrics(mf_sub['Actual Units'].values, mf_sub['10-Day Forecast'].values)

    # Platform totals
    plat_totals = {
        'Amazon Predicted': int(df_out['Amazon Predicted'].sum()),
        'eBay Predicted': int(df_out['eBay Predicted'].sum()),
        'Website Predicted': int(df_out['Website Predicted'].sum()),
        'Other Predicted': int(df_out['Other Predicted'].sum()),
        '10-Day Forecast Total': int(df_out['10-Day Forecast'].sum()),
        'Actual Units Total': int(df_out['Actual Units'].sum())
    }

    results = {
        "combined": comb_metrics,
        "rimmel": rim_metrics,
        "max_factor": mf_metrics,
        "platform_totals": plat_totals,
        "sku_counts": {
            "total": len(df_out),
            "rimmel": len(rim_sub),
            "max_factor": len(mf_sub)
        }
    }

    metrics_json_path = os.path.join(PIPELINE_ROOT, "reports", "phase4a_validation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Metrics summary:")
    logger.info(f"Combined: Actual={comb_metrics['actual']:,}, Pred={comb_metrics['predicted']:,}, WAPE={comb_metrics['wape']}%, Bias={comb_metrics['bias']}%")
    logger.info(f"Rimmel:   Actual={rim_metrics['actual']:,}, Pred={rim_metrics['predicted']:,}, WAPE={rim_metrics['wape']}%, Bias={rim_metrics['bias']}%")
    logger.info(f"Max Factor: Actual={mf_metrics['actual']:,}, Pred={mf_metrics['predicted']:,}, WAPE={mf_metrics['wape']}%, Bias={mf_metrics['bias']}%")

if __name__ == "__main__":
    main()
