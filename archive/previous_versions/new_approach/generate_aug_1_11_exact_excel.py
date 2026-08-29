"""
Generate 1 to 11 August 2026 Exact Format Excel Forecast Report
Exact Columns:
1. order_sku
2. date ("1 to 11")
3. predicted_sales (Integer, no decimals)
4. lower_bound (Integer, no decimals)
5. upper_bound (Integer, no decimals)
"""
import os
import pandas as pd
import numpy as np

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..')
    forecast_csv_path = os.path.join(base_dir, 'output_results', 'global_588_august_forecast.csv')
    
    print("Loading forecast data...")
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
        'predicted_sales': np.round(agg_df['predicted_sales'], 0).astype(int),
        'lower_bound': np.round(agg_df['lower_bound_95'], 0).astype(int),
        'upper_bound': np.round(agg_df['upper_bound_95'], 0).astype(int)
    })
    
    # Sort by predicted_sales descending
    report_df = report_df.sort_values(by='predicted_sales', ascending=False).reset_index(drop=True)
    
    out_excel_1 = os.path.join(project_dir, 'Rimmel_588_SKUs_Forecast_1_to_11.xlsx')
    out_excel_2 = os.path.join(base_dir, 'output_results', 'Rimmel_588_SKUs_Forecast_1_to_11.xlsx')
    
    for out_path in [out_excel_1, out_excel_2]:
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, sheet_name='Forecast_1_to_11', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Forecast_1_to_11']
            
            # Format header
            header_format = workbook.add_format({
                'bold': True,
                'fg_color': '#1E293B',
                'font_color': '#FFFFFF',
                'border': 1
            })
            
            for col_num, value in enumerate(report_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Set column widths
            worksheet.set_column('A:A', 25) # order_sku
            worksheet.set_column('B:B', 15) # date
            worksheet.set_column('C:C', 18) # predicted_sales
            worksheet.set_column('D:D', 16) # lower_bound
            worksheet.set_column('E:E', 16) # upper_bound

    print(f"Successfully generated exact Excel report for 588 SKUs ({len(report_df)} rows)!")
    print(f"Saved to: {out_excel_1}")
    print(f"Sample Rows:")
    print(report_df.head(10).to_string(index=False))

if __name__ == '__main__':
    generate_report()
