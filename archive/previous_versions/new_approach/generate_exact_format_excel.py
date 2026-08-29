"""
Generate Exact Format Excel Forecast Report
Columns:
- order_sku
- date ("1 to 11")
- predicted price (Integer, no decimals)
- lower bound (Integer, no decimals)
- upper bound (Integer, no decimals)
"""
import os
import pandas as pd
import numpy as np

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..')
    forecast_csv_path = os.path.join(base_dir, 'output_results', 'global_588_august_forecast.csv')
    
    df_f = pd.read_csv(forecast_csv_path)
    df_f['date'] = pd.to_datetime(df_f['date'])
    
    # Filter 1 August to 11 August 2026
    df_sub = df_f[(df_f['date'] >= '2026-08-01') & (df_f['date'] <= '2026-08-11')].copy()
    
    # Aggregate predictions for 11 days per SKU
    agg_df = df_sub.groupby('order_item_sku').agg({
        'predicted_sales': 'sum',
        'lower_bound_95': 'sum',
        'upper_bound_95': 'sum'
    }).reset_index()
    
    # Format exact requested columns
    report_df = pd.DataFrame({
        'order_sku': agg_df['order_item_sku'],
        'date': '1 to 11',
        'predicted price': np.round(agg_df['predicted_sales'], 0).astype(int),
        'lower bound': np.round(agg_df['lower_bound_95'], 0).astype(int),
        'upper bound': np.round(agg_df['upper_bound_95'], 0).astype(int)
    })
    
    # Sort by predicted price descending
    report_df = report_df.sort_values(by='predicted price', ascending=False).reset_index(drop=True)
    
    out_excel = os.path.join(project_dir, 'Rimmel_588_SKUs_Forecast_1_to_11.xlsx')
    
    with pd.ExcelWriter(out_excel, engine='xlsxwriter') as writer:
        report_df.to_excel(writer, sheet_name='Forecast', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Forecast']
        
        header_format = workbook.add_format({
            'bold': True,
            'fg_color': '#1E293B',
            'font_color': '#FFFFFF',
            'border': 1
        })
        
        for col_num, value in enumerate(report_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 15)
        worksheet.set_column('C:C', 18)
        worksheet.set_column('D:D', 16)
        worksheet.set_column('E:E', 16)

    print(f"Generated exact format Excel report successfully at {out_excel}")

if __name__ == '__main__':
    generate_report()
