"""
REPORT GENERATOR MODULE
========================
Generates client-ready, multi-sheet formatted Excel deliverables
with dynamic date periods, clear inventory actions, and robust error handling.
"""
import os
import pandas as pd

def generate_client_excel_report(forecast_dataframe, output_filepath, is_evaluation=False):
    """
    Generates a formatted multi-sheet Excel report for executive review.
    
    Args:
        forecast_dataframe (pd.DataFrame): Output dataframe from dynamic forecasting engine.
        output_filepath (str): Target output filepath (.xlsx).
        is_evaluation (bool): If True, populates Actual Sales and error diagnostics.
                              If False, leaves Actual Sales blank for client entry.
                              
    Returns:
        str: Absolute path of the successfully saved report.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    df_report = forecast_dataframe.copy()
    
    # ── Sheet 1: Executive Forecast & Planning ─────────────────────────────────
    if is_evaluation:
        main_columns = [
            'Date', 'Product SKU', 'Category', 'Actual Sales',
            'Baseline Prediction', 'Momentum Prediction', 'Recommended Forecast',
            'Recommended Error', 'Model Performance Comparison',
            'Confidence', 'Risk / Status', 'Reason'
        ]
    else:
        if 'Actual Sales' not in df_report.columns:
            df_report['Actual Sales'] = ""
        main_columns = [
            'Date', 'Product SKU', 'Category', 'Actual Sales',
            'Baseline Prediction', 'Momentum Prediction', 'Recommended Forecast',
            'Confidence', 'Risk / Status', 'Reason',
            'Current Stock (Units)', 'Estimated Days of Inventory', 'Inventory Health Status'
        ]
        
    df_main_sheet = df_report[[c for c in main_columns if c in df_report.columns]].copy()
    
    # ── Sheet 2: Inventory Action Summary ──────────────────────────────────────
    inv_columns = [
        'Product SKU', 'Category', 'Current Stock (Units)', 'Selling Price ($)',
        'Recommended Daily Demand', 'Estimated Days of Inventory',
        'Inventory Health Status', 'Inventory Action Recommendation', 'Annual Sales (Units)'
    ]
    df_inv_sheet = df_report[[c for c in inv_columns if c in df_report.columns]].copy()
    if 'Current Stock (Units)' in df_inv_sheet.columns and 'Selling Price ($)' in df_inv_sheet.columns:
        df_inv_sheet['Trapped Working Capital ($)'] = round(df_inv_sheet['Current Stock (Units)'] * df_inv_sheet['Selling Price ($)'], 2)
        
    # Safe write with fallback if file is locked
    target_path = output_filepath
    try:
        writer = pd.ExcelWriter(target_path, engine='xlsxwriter')
    except PermissionError:
        base, ext = os.path.splitext(output_filepath)
        target_path = f"{base}_new{ext}"
        writer = pd.ExcelWriter(target_path, engine='xlsxwriter')
        
    sheet_name_1 = 'Holdout Evaluation' if is_evaluation else 'Executive Forecast'
    df_main_sheet.to_excel(writer, sheet_name=sheet_name_1, index=False)
    df_inv_sheet.to_excel(writer, sheet_name='Inventory Action Summary', index=False)
    
    wb = writer.book
    hdr_blue  = wb.add_format({'bold': True, 'bg_color': '#1f497d', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
    hdr_act   = wb.add_format({'bold': True, 'bg_color': '#2e75b6', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
    hdr_rec   = wb.add_format({'bold': True, 'bg_color': '#238636', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
    hdr_warn  = wb.add_format({'bold': True, 'bg_color': '#d29922', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
    hdr_risk  = wb.add_format({'bold': True, 'bg_color': '#b71c1c', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
    
    ws1 = writer.sheets[sheet_name_1]
    for i, col in enumerate(df_main_sheet.columns):
        if 'Actual' in col:
            ws1.write(0, i, col, hdr_act)
        elif 'Recommended Forecast' in col:
            ws1.write(0, i, col, hdr_rec)
        elif 'Confidence' in col:
            ws1.write(0, i, col, hdr_warn)
        elif 'Risk' in col or 'Error' in col:
            ws1.write(0, i, col, hdr_risk)
        else:
            ws1.write(0, i, col, hdr_blue)
            
    ws1.set_column('A:A', 24)
    ws1.set_column('B:C', 24)
    ws1.set_column('D:H', 18)
    ws1.set_column('I:K', 18)
    ws1.set_column('L:L', 75)
    ws1.freeze_panes(1, 0)
    
    ws2 = writer.sheets['Inventory Action Summary']
    ws2.set_column('A:B', 24)
    ws2.set_column('C:F', 20)
    ws2.set_column('G:H', 32)
    ws2.set_column('I:J', 20)
    ws2.freeze_panes(1, 0)
    
    writer.close()
    return target_path
