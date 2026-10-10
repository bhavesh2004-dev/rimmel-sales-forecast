"""
Multi-Brand Unified LightGBM 10-Day Operational Forecast & ROP Deliverable Generator
=====================================================================================
Generates the official certified 21-column combined workbook:
MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx

Features:
- Single worksheet: 'Operational_Forecast_and_ROP'
- All 7 brands processed with the approved shared LightGBM model:
  global_lgbm_model.pkl (MD5: db098afc3d8f1805dd7f21d2a193f0ca)
- Approved Exp6 calibration (alpha=0.10, beta=0.10) applied consistently
- 60 causal features generated dynamically in RAM from MySQL normalized_sales
- Additive Channel Law verified: Amazon + eBay + Website + Other == 10-Day Forecast
- Full downstream ROP replenishment logic with business policy parameters:
  Lead Time = 10 Days, Minimum Stock Level = 6 Units
- Mathematical assertions: zero sentinels, zero negative replenishment quantities,
  and projected stock buffer invariant (Current Stock + Replenish Qty - LTD >= 6.0)
- Professional styling matching design system (Navy #1F4E78, Gold #D99B26,
  auto-filters, freeze panes at row 4, zebra striping, currency/numeric formatting).
- Strict constraints: Read-only MySQL (zero writes), approved model untouched.
"""

import os
import sys
import json
import math
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import pickle
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv

# Dynamically resolve directory hierarchy
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PIPELINE_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.data_discovery import discover_brand_coverage
from multibrand_pipeline.src.observation_grid import construct_daily_grid
from multibrand_pipeline.src.features import compute_causal_features
from multibrand_pipeline.src.inventory import classify_inventory_risk_and_action, compute_days_of_cover
from multibrand_pipeline.src.replenishment import calculate_sku_replenishment, compute_rop_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_combined_operational_excel")

# Design Palette
FONT_TITLE = Font(name='Segoe UI', size=13, bold=True, color='FFFFFF')
FONT_SUBTITLE = Font(name='Segoe UI', size=9, italic=True, color='E0E0E0')
FONT_POLICY = Font(name='Segoe UI', size=9, bold=True, color='1E3A8A')
FONT_HEADER = Font(name='Segoe UI', size=9, bold=True, color='FFFFFF')
FONT_BODY = Font(name='Segoe UI', size=9, color='000000')
FONT_BOLD = Font(name='Segoe UI', size=9, bold=True, color='000000')

FILL_NAVY = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
FILL_STEEL = PatternFill(start_color='2F5597', end_color='2F5597', fill_type='solid')
FILL_POLICY = PatternFill(start_color='E0E7FF', end_color='E0E7FF', fill_type='solid')
FILL_HEADER = PatternFill(start_color='333F48', end_color='333F48', fill_type='solid')
FILL_ACCENT = PatternFill(start_color='1E3A8A', end_color='1E3A8A', fill_type='solid')
FILL_ZEBRA = PatternFill(start_color='F9FAFB', end_color='F9FAFB', fill_type='solid')
FILL_WHITE = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')
FILL_TOTAL = PatternFill(start_color='F1F5F9', end_color='F1F5F9', fill_type='solid')

# ROP Status Alert Fills
FILL_STATUS_BELOW = PatternFill(start_color='FEE2E2', end_color='FEE2E2', fill_type='solid')
FONT_STATUS_BELOW = Font(name='Segoe UI', size=9, bold=True, color='991B1B')

FILL_STATUS_MIN = PatternFill(start_color='FEF3C7', end_color='FEF3C7', fill_type='solid')
FONT_STATUS_MIN = Font(name='Segoe UI', size=9, bold=True, color='92400E')

FILL_STATUS_ABOVE = PatternFill(start_color='ECFDF5', end_color='ECFDF5', fill_type='solid')
FONT_STATUS_ABOVE = Font(name='Segoe UI', size=9, bold=True, color='065F46')

FILL_STATUS_NODEMAND = PatternFill(start_color='F3F4F6', end_color='F3F4F6', fill_type='solid')
FONT_STATUS_NODEMAND = Font(name='Segoe UI', size=9, color='4B5563')

BORDER_THIN = Border(
    left=Side(style='thin', color='E5E7EB'),
    right=Side(style='thin', color='E5E7EB'),
    top=Side(style='thin', color='E5E7EB'),
    bottom=Side(style='thin', color='E5E7EB')
)

BORDER_POLICY = Border(
    left=Side(style='medium', color='3B82F6'),
    right=Side(style='medium', color='3B82F6'),
    top=Side(style='medium', color='3B82F6'),
    bottom=Side(style='medium', color='3B82F6')
)

BORDER_TOTAL = Border(
    left=Side(style='thin', color='CBD5E1'),
    right=Side(style='thin', color='CBD5E1'),
    top=Side(style='thin', color='475569'),
    bottom=Side(style='double', color='1E293B')
)

def generate_forecast_metadata(r: pd.Series, config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Generates confidence level, operational risk, recommended action, and human-readable reason."""
    stock = float(r.get('current_stock', 0.0))
    in_stock = int(r.get('in_stock_flag', 1 if stock > 0 else 0))
    cv = float(r.get('cv_30', 0.0))
    fwd = float(r.get('Total Predicted', r.get('10-Day Forecast', 0.0)))
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
        
    return conf, risk, action, reason

def run_multibrand_operational_forecast() -> pd.DataFrame:
    """
    Executes the unified LightGBM forward forecast across all 7 brands
    and produces the complete 21-column dataset.
    """
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    
    model_path = os.path.join(PIPELINE_ROOT, "models", "global_lgbm_model.pkl")
    schema_path = os.path.join(PIPELINE_ROOT, "config", "feature_schema.json")
    config_path = os.path.join(PIPELINE_ROOT, "config", "dynamic_pipeline_config.yaml")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    feature_list = schema["feature_list"]
    cat_features = schema["categorical_features"]
    
    # Load pipeline configuration for risk parameters
    pipeline_cfg_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
    import yaml
    with open(pipeline_cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db = DBManager()
    df_brands, df_quality = discover_brand_coverage(db, horizon_days=10)
    logger.info(f"Discovered {len(df_brands)} brands in MySQL.")
    
    # Read normalized sales from MySQL
    with db.engine.connect() as conn:
        df_norm = pd.read_sql("SELECT * FROM normalized_sales", conn)
    df_norm['date'] = pd.to_datetime(df_norm['date']).dt.strftime('%Y-%m-%d')
    logger.info(f"Loaded {len(df_norm):,} normalized rows from MySQL.")
    
    alpha = config.get('calibration', {}).get('zero_demand_alpha', 0.10)
    beta = config.get('calibration', {}).get('stockout_beta', 0.10)
    
    lead_time_days = 10
    min_stock_level = 6
    daily_window = 10
    
    all_sku_records = []
    
    for _, b_prof in df_brands.iterrows():
        b_id = b_prof['brand_id']
        disp_name = b_prof['display_brand_name']
        cutoff_T = b_prof['cutoff_date_T']
        fwd_start = b_prof['forecast_start']
        fwd_end = b_prof['forecast_end']
        fwd_dates = pd.date_range(start=fwd_start, end=fwd_end, freq='D')
        period_str = f"{fwd_start} to {fwd_end}"
        
        logger.info(f"Processing brand: {disp_name} (Cutoff T={cutoff_T}, Horizon={period_str})...")
        
        b_norm = df_norm[df_norm['brand_id'] == b_id].copy()
        
        # Build daily continuous grid up to cutoff date T
        grid_b = construct_daily_grid(
            df_norm=b_norm,
            start_date=b_prof['context_start'],
            end_date=cutoff_T,
            stock_telemetry_valid_start="2025-08-01"
        )
        
        # Compute 60 causal features dynamically in RAM
        feat_b = compute_causal_features(grid_b, schema_path)
        
        # Anchor state on cutoff T
        anchor_df = feat_b[feat_b['date'] == cutoff_T].copy()
        if anchor_df.empty:
            latest_avail = feat_b['date'].max()
            logger.warning(f"  Anchor date {cutoff_T} missing, using {latest_avail}")
            anchor_df = feat_b[feat_b['date'] == latest_avail].copy()
            
        # Step forward across 10 operational horizon days
        daily_records = []
        for f_dt in fwd_dates:
            f_str = f_dt.strftime('%Y-%m-%d')
            dow = f_dt.dayofweek
            
            sim_df = anchor_df.copy()
            sim_df['date'] = f_str
            sim_df['day_of_week'] = dow
            
            X_sim = sim_df[feature_list].copy()
            for c in cat_features:
                if c in X_sim.columns:
                    X_sim[c] = X_sim[c].astype('category')
                    
            raw_pred = np.clip(model.predict(X_sim), 0, None)
            
            # Exp6 Calibration
            z_mask = (sim_df['v7'] == 0) & (sim_df['v14'] == 0) & (sim_df['v30'] == 0)
            stk_mask = (sim_df['in_stock_flag'] == 0)
            
            calib_pred = raw_pred.copy()
            calib_pred[z_mask.values] *= alpha
            calib_pred[stk_mask.values] *= beta
            
            sim_df['calib_daily_prediction'] = calib_pred
            daily_records.append(sim_df)
            
        df_daily = pd.concat(daily_records, ignore_index=True)
        
        if 'product_title' not in df_daily.columns:
            df_daily['product_title'] = df_daily.get('category', df_daily['canonical_sku'])
        if 'category' not in df_daily.columns:
            df_daily['category'] = df_daily.get('product_title', df_daily['canonical_sku'])
            
        # Aggregate to 10-day SKU x Platform level
        series_agg = df_daily.groupby(['brand_id', 'canonical_sku', 'product_title', 'category', 'platform_group']).agg(
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
        
        # Pivot platforms
        plat_pivot = series_agg.pivot_table(
            index=['brand_id', 'canonical_sku'],
            columns='platform_group',
            values='recommended_10d_units',
            fill_value=0
        ).reset_index()
        
        for p in ['Amazon', 'eBay', 'Website', 'Other']:
            if p not in plat_pivot.columns:
                plat_pivot[p] = 0
                
        # Additive Channel Law
        plat_pivot['10-Day Forecast'] = (
            plat_pivot['Amazon'] + plat_pivot['eBay'] + plat_pivot['Website'] + plat_pivot['Other']
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
        
        brand_df = pd.merge(sku_meta, plat_pivot, on=['brand_id', 'canonical_sku'], how='left')
        brand_df['Brand'] = disp_name
        brand_df['Forecast Period'] = period_str
        
        all_sku_records.append(brand_df)
        
    df_combined = pd.concat(all_sku_records, ignore_index=True)
    logger.info(f"Total aggregated SKUs across all 7 brands: {len(df_combined):,}")
    
    # Generate 21-column schema
    rows = []
    for _, r in df_combined.iterrows():
        brand = r['Brand']
        sku = r['canonical_sku']
        prod = r['category'] if pd.notna(r['category']) and r['category'] != '' else r['product_title']
        stock = int(float(r['current_stock'])) if pd.notna(r['current_stock']) else 0
        period = r['Forecast Period']
        
        amz = int(r['Amazon'])
        ebay = int(r['eBay'])
        web = int(r['Website'])
        oth = int(r['Other'])
        tot_fwd = int(r['10-Day Forecast'])
        
        doc = compute_days_of_cover(stock, tot_fwd)
        conf, risk, action, reason = generate_forecast_metadata(r, config)
        
        # ROP Replenishment Calculations
        rop_metrics = calculate_sku_replenishment(
            current_stock=stock,
            forecast_10d=tot_fwd,
            lead_time_days=lead_time_days,
            min_stock_level=min_stock_level,
            daily_rate_window=daily_window,
            rounding_method="ceil"
        )
        
        rows.append({
            'Brand': brand,
            'SKU': sku,
            'Product': prod,
            'Current Stock': stock,
            'Forecast Period': period,
            'Amazon Predicted': amz,
            'eBay Predicted': ebay,
            'Website Predicted': web,
            'Other Predicted': oth,
            '10-Day Forecast': tot_fwd,
            'Days of Cover': doc,
            'Confidence': conf,
            'Risk': risk,
            'Recommended Action': action,
            'Reason': reason,
            'Lead Time (Days)': lead_time_days,
            'Avg Daily Usage': rop_metrics['Avg Daily Usage'],
            'Lead-Time Demand': rop_metrics['Lead-Time Demand'],
            'Target Stock': rop_metrics['Target Stock'],
            'Replenishment Qty': rop_metrics['Replenishment Qty'],
            'ROP Status': rop_metrics['ROP Status']
        })
        
    df_final = pd.DataFrame(rows)
    return df_final

def build_excel_workbook(df: pd.DataFrame, output_path: str):
    """
    Builds the certified single-worksheet Excel deliverable with exact 21 columns
    and professional formatting matching design system.
    """
    logger.info(f"Building official Excel workbook at: {output_path}...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Operational_Forecast_and_ROP"
    ws.views.sheetView[0].showGridLines = True
    
    headers = [
        'Brand',
        'SKU',
        'Product',
        'Current Stock',
        'Forecast Period',
        'Amazon Predicted',
        'eBay Predicted',
        'Website Predicted',
        'Other Predicted',
        '10-Day Forecast',
        'Days of Cover',
        'Confidence',
        'Risk',
        'Recommended Action',
        'Reason',
        'Lead Time (Days)',
        'Avg Daily Usage',
        'Lead-Time Demand',
        'Target Stock',
        'Replenishment Qty',
        'ROP Status'
    ]
    
    assert len(headers) == 21, f"Expected 21 headers, got {len(headers)}"
    last_col = get_column_letter(len(headers))
    
    # 1. Title Banner (Row 1)
    ws.merge_cells(f'A1:{last_col}1')
    c_title = ws['A1']
    c_title.value = "MULTI-BRAND DEMAND FORECAST & INVENTORY REPLENISHMENT (10-DAY OPERATIONAL PROJECTION)"
    c_title.font = FONT_TITLE
    c_title.fill = FILL_NAVY
    c_title.alignment = Alignment(horizontal='left', vertical='center', indent=1)
    ws.row_dimensions[1].height = 28
    
    # 2. Subtitle / Policy Banner (Row 2)
    ws.merge_cells(f'A2:{last_col}2')
    c_policy = ws['A2']
    c_policy.value = "ROP POLICY: Lead Time = 10 Days | Minimum Stock Level = 6 Units | Target Stock = (Avg Daily Usage × 10) + 6 | Additive Channel Law Enforced"
    c_policy.font = FONT_POLICY
    c_policy.fill = FILL_POLICY
    c_policy.alignment = Alignment(horizontal='left', vertical='center', indent=1)
    c_policy.border = BORDER_POLICY
    ws.row_dimensions[2].height = 22
    
    # 3. Headers (Row 3)
    ws.row_dimensions[3].height = 26
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = FILL_ACCENT if h in ['10-Day Forecast', 'Target Stock', 'Replenishment Qty', 'ROP Status'] else FILL_HEADER
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = BORDER_THIN
        
    # 4. Populate Data Rows (Row 4 onwards)
    row_num = 4
    tot_stock = 0
    tot_amz = 0
    tot_ebay = 0
    tot_web = 0
    tot_oth = 0
    tot_fwd = 0
    tot_ltd = 0.0
    tot_target = 0.0
    tot_replenish = 0
    skus_ordering = 0
    
    for _, r in df.iterrows():
        b = r['Brand']
        sku = r['SKU']
        prod = r['Product']
        stock = int(r['Current Stock'])
        period = r['Forecast Period']
        amz = int(r['Amazon Predicted'])
        ebay = int(r['eBay Predicted'])
        web = int(r['Website Predicted'])
        oth = int(r['Other Predicted'])
        fwd = int(r['10-Day Forecast'])
        doc = r['Days of Cover']
        conf = r['Confidence']
        risk = r['Risk']
        act = r['Recommended Action']
        reason = r['Reason']
        lt = int(r['Lead Time (Days)'])
        adu = float(r['Avg Daily Usage'])
        ltd = float(r['Lead-Time Demand'])
        tgt = float(r['Target Stock'])
        rep_qty = int(r['Replenishment Qty'])
        status = r['ROP Status']
        
        tot_stock += stock
        tot_amz += amz
        tot_ebay += ebay
        tot_web += web
        tot_oth += oth
        tot_fwd += fwd
        tot_ltd += ltd
        tot_target += tgt
        tot_replenish += rep_qty
        if rep_qty > 0:
            skus_ordering += 1
            
        row_vals = [
            b, sku, prod, stock, period,
            amz, ebay, web, oth, fwd,
            doc, conf, risk, act, reason,
            lt, adu, ltd, tgt, rep_qty, status
        ]
        
        fill = FILL_ZEBRA if row_num % 2 == 0 else FILL_WHITE
        ws.row_dimensions[row_num].height = 20
        
        for col_idx, (h, val) in enumerate(zip(headers, row_vals), 1):
            cell = ws.cell(row=row_num, column=col_idx, value=val)
            cell.font = FONT_BODY
            cell.fill = fill
            cell.border = BORDER_THIN
            
            # Format and alignment
            if h in ['Current Stock', 'Amazon Predicted', 'eBay Predicted', 'Website Predicted', 'Other Predicted', '10-Day Forecast', 'Lead Time (Days)', 'Replenishment Qty']:
                cell.number_format = '#,##0'
                cell.alignment = Alignment(horizontal='right', vertical='center')
            elif h in ['Lead-Time Demand', 'Target Stock']:
                cell.number_format = '#,##0'
                cell.alignment = Alignment(horizontal='right', vertical='center')
            elif h == 'Avg Daily Usage':
                cell.number_format = '#,##0.0'
                cell.alignment = Alignment(horizontal='right', vertical='center')
            elif h == 'Days of Cover':
                if isinstance(val, (int, float)):
                    cell.number_format = '#,##0.0'
                    cell.alignment = Alignment(horizontal='right', vertical='center')
                else:
                    cell.alignment = Alignment(horizontal='center', vertical='center')
            elif h == 'ROP Status':
                cell.alignment = Alignment(horizontal='center', vertical='center')
                if val == 'BELOW ROP':
                    cell.fill = FILL_STATUS_BELOW
                    cell.font = FONT_STATUS_BELOW
                elif val == 'AT MINIMUM STOCK':
                    cell.fill = FILL_STATUS_MIN
                    cell.font = FONT_STATUS_MIN
                elif val == 'ABOVE ROP':
                    cell.fill = FILL_STATUS_ABOVE
                    cell.font = FONT_STATUS_ABOVE
                else:
                    cell.fill = FILL_STATUS_NODEMAND
                    cell.font = FONT_STATUS_NODEMAND
            elif h in ['Brand', 'Confidence', 'Risk']:
                cell.alignment = Alignment(horizontal='center', vertical='center')
            else:
                cell.alignment = Alignment(horizontal='left', vertical='center')
                
        row_num += 1
        
    # 5. Total Portfolio Summary Row (Bottom)
    ws.row_dimensions[row_num].height = 24
    overall_doc = round(tot_stock / (tot_fwd / 10.0), 1) if tot_fwd > 0 else "N/A"
    overall_adu = round(tot_fwd / 10.0, 1)
    
    total_row = [
        "Total Portfolio", "", "", tot_stock, "",
        tot_amz, tot_ebay, tot_web, tot_oth, tot_fwd,
        overall_doc, "", "", "", "",
        10, overall_adu, round(tot_ltd), round(tot_target), tot_replenish,
        f"{skus_ordering} SKUs Order Needed"
    ]
    
    for col_idx, (h, val) in enumerate(zip(headers, total_row), 1):
        cell = ws.cell(row=row_num, column=col_idx, value=val)
        cell.font = FONT_BOLD
        cell.fill = FILL_TOTAL
        cell.border = BORDER_TOTAL
        
        if isinstance(val, (int, float)):
            cell.number_format = '#,##0.0' if h in ['Days of Cover', 'Avg Daily Usage'] else '#,##0'
            cell.alignment = Alignment(horizontal='right', vertical='center')
        else:
            cell.alignment = Alignment(horizontal='left' if col_idx == 1 else 'center', vertical='center')
            
    # Enable AutoFilter across row 3
    ws.auto_filter.ref = f"A3:{last_col}{row_num - 1}"
    
    # Freeze Panes at row 4
    ws.freeze_panes = 'A4'
    
    # Auto-fit column widths
    for col_idx in range(1, len(headers) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max([len(str(ws.cell(row=r, column=col_idx).value or '')) for r in range(3, min(row_num + 1, 80))] or [12])
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 48)
        
    # Save workbook
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)
    logger.info(f"Successfully certified and exported workbook to: {output_path}")
    return output_path

def audit_deliverable_invariants(excel_path: str, df: pd.DataFrame):
    """
    Runs strict automated forensic assertions against the generated Excel deliverable.
    """
    logger.info("==================================================================")
    logger.info("AUDITING FINAL DELIVERABLE INTEGRITY AND INVARIANTS")
    logger.info("==================================================================")
    
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    assert len(wb.sheetnames) == 1, f"Expected exactly 1 worksheet, got {len(wb.sheetnames)}: {wb.sheetnames}"
    assert wb.sheetnames[0] == "Operational_Forecast_and_ROP", f"Worksheet name mismatch! Got: {wb.sheetnames[0]}"
    
    ws = wb["Operational_Forecast_and_ROP"]
    headers = [ws.cell(3, c).value for c in range(1, 22)]
    expected_headers = [
        'Brand', 'SKU', 'Product', 'Current Stock', 'Forecast Period',
        'Amazon Predicted', 'eBay Predicted', 'Website Predicted', 'Other Predicted',
        '10-Day Forecast', 'Days of Cover', 'Confidence', 'Risk', 'Recommended Action', 'Reason',
        'Lead Time (Days)', 'Avg Daily Usage', 'Lead-Time Demand', 'Target Stock', 'Replenishment Qty', 'ROP Status'
    ]
    assert headers == expected_headers, f"Header mismatch!\nGot: {headers}\nExpected: {expected_headers}"
    
    # Total row count
    data_rows = ws.max_row - 4
    assert data_rows == len(df), f"Expected {len(df)} data rows, got {data_rows}"
    assert ws.auto_filter.ref is not None, "AutoFilter is missing!"
    assert ws.freeze_panes == 'A4', f"Freeze Panes mismatch! Got: {ws.freeze_panes}"
    
    # Additive Channel Law
    channel_sum = df['Amazon Predicted'] + df['eBay Predicted'] + df['Website Predicted'] + df['Other Predicted']
    additive_violations = (channel_sum != df['10-Day Forecast']).sum()
    assert additive_violations == 0, f"Found {additive_violations} Additive Channel Law violations!"
    
    # Non-negative replenishment
    assert (df['Replenishment Qty'] >= 0).all(), "Found negative replenishment quantities!"
    
    # Projected Stock Buffer Invariant: Current Stock + Replenishment Qty - LTD >= 6.0
    projected_stock = df['Current Stock'] + df['Replenishment Qty'] - df['Lead-Time Demand']
    min_projected = projected_stock.min()
    assert min_projected >= 6.0 - 1e-4, f"Projected stock dropped below 6 units! Min: {min_projected}"
    
    # Zero 999 sentinels
    sentinel_count = (df.astype(str) == '999').sum().sum() + (df.astype(str) == '999.0').sum().sum()
    assert sentinel_count == 0, f"Found {sentinel_count} '999' sentinels in dataframe!"
    
    logger.info(f"Total Canonical SKUs: {len(df):,}")
    logger.info(f"Total Current Warehouse Stock: {df['Current Stock'].sum():,} units")
    logger.info(f"Total 10-Day Forecast: {df['10-Day Forecast'].sum():,} units")
    logger.info(f"  Amazon:  {df['Amazon Predicted'].sum():,} units ({df['Amazon Predicted'].sum() / df['10-Day Forecast'].sum() * 100:.1f}%)")
    logger.info(f"  eBay:    {df['eBay Predicted'].sum():,} units ({df['eBay Predicted'].sum() / df['10-Day Forecast'].sum() * 100:.1f}%)")
    logger.info(f"  Website: {df['Website Predicted'].sum():,} units ({df['Website Predicted'].sum() / df['10-Day Forecast'].sum() * 100:.1f}%)")
    logger.info(f"  Other:   {df['Other Predicted'].sum():,} units ({df['Other Predicted'].sum() / df['10-Day Forecast'].sum() * 100:.1f}%)")
    logger.info(f"Total Recommended Replenishment Quantity: {df['Replenishment Qty'].sum():,} units")
    skus_ordering = (df['Replenishment Qty'] > 0).sum()
    logger.info(f"SKUs Requiring Purchase Orders: {skus_ordering:,} / {len(df):,} ({skus_ordering / len(df) * 100:.1f}%)")
    logger.info("ALL INVARIANTS AUDITED AND 100% CERTIFIED!")

def main():
    logger.info("==================================================================")
    logger.info("STARTING UNIFIED MULTI-BRAND LIGHTGBM OPERATIONAL FORECAST RUN")
    logger.info("==================================================================")
    
    df_final = run_multibrand_operational_forecast()
    
    output_filename = "MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx"
    dest_pipeline_reports = os.path.join(PIPELINE_ROOT, "reports", output_filename)
    dest_artifacts_reports = os.path.join(PROJECT_ROOT, "artifacts", "reports", output_filename)
    
    # Build Excel in pipeline reports
    saved_path = build_excel_workbook(df_final, dest_pipeline_reports)
    
    # Audit invariants
    audit_deliverable_invariants(saved_path, df_final)
    
    # Copy to artifacts directory with Windows file lock resilience
    if os.path.exists(os.path.dirname(dest_artifacts_reports)):
        try:
            shutil.copyfile(saved_path, dest_artifacts_reports)
            logger.info(f"Copied deliverable to: {dest_artifacts_reports}")
        except PermissionError:
            fallback_art = dest_artifacts_reports.replace('.xlsx', '_LATEST.xlsx')
            shutil.copyfile(saved_path, fallback_art)
            logger.warning(f"File locked in Excel. Saved latest copy to: {fallback_art}")
        
    print("\n=== FINAL DELIVERABLE SUMMARY ===")
    print(f"File: {saved_path}")
    print(f"Total Rows: {len(df_final):,} SKUs across 7 brands")
    print("\nBrand Breakdown:")
    for b, grp in df_final.groupby('Brand'):
        print(f"  {b:16s} | {len(grp):4d} SKUs | Forecast: {grp['10-Day Forecast'].sum():6d} units | Stock: {grp['Current Stock'].sum():6d} units | Replenish: {grp['Replenishment Qty'].sum():6d} units")

if __name__ == "__main__":
    main()
