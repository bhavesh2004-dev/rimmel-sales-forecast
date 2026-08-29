"""
Generate 1 August to 11 August 2026 Excel Forecast Report for 588 SKUs
Columns:
- Product Name (Title)
- SKU Code
- Category
- Date Period ("1 Aug 2026 - 11 Aug 2026")
- Predicted Value (Total 11-day forecast units)
- Lower Bound (Total 11-day lower 95% confidence bound)
- Upper Bound (Total 11-day upper 95% safety stock buffer)
"""
import os
import pandas as pd
import numpy as np

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..')
    excel_raw_path = os.path.join(project_dir, 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
    forecast_csv_path = os.path.join(base_dir, 'output_results', 'global_588_august_forecast.csv')
    
    print("Loading forecast data and raw product metadata...")
    df_f = pd.read_csv(forecast_csv_path)
    df_f['date'] = pd.to_datetime(df_f['date'])
    
    # Filter 1 August to 11 August 2026
    df_sub = df_f[(df_f['date'] >= '2026-08-01') & (df_f['date'] <= '2026-08-11')].copy()
    
    # Load raw file for product titles / names
    df_raw = pd.read_excel(excel_raw_path)
    sku_title_map = df_raw.groupby('sku')['title'].first().to_dict() if 'title' in df_raw.columns else {}
    
    # Aggregate predictions for 11 days per SKU
    agg_df = df_sub.groupby(['order_item_sku', 'category']).agg({
        'predicted_sales': 'sum',
        'lower_bound_95': 'sum',
        'upper_bound_95': 'sum'
    }).reset_index()
    
    agg_df['Product Name'] = agg_df['order_item_sku'].map(sku_title_map).fillna(agg_df['order_item_sku'])
    agg_df['Date Period'] = '1 Aug 2026 - 11 Aug 2026'
    
    # Format columns in requested order
    report_df = pd.DataFrame({
        'Product Name': agg_df['Product Name'],
        'SKU Code': agg_df['order_item_sku'],
        'Category': agg_df['category'],
        'Date Period': agg_df['Date Period'],
        'Predicted Value': np.round(agg_df['predicted_sales'], 0).astype(int),
        'Lower Bound': np.round(agg_df['lower_bound_95'], 0).astype(int),
        'Upper Bound': np.round(agg_df['upper_bound_95'], 0).astype(int)
    })
    
    # Sort by Predicted Value descending
    report_df = report_df.sort_values(by='Predicted Value', ascending=False).reset_index(drop=True)
    
    out_excel_1 = os.path.join(project_dir, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
    out_excel_2 = os.path.join(base_dir, 'output_results', 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
    
    for out_path in [out_excel_1, out_excel_2]:
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, sheet_name='1Aug_to_11Aug_Forecast', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['1Aug_to_11Aug_Forecast']
            
            # Format header & columns
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1E293B',
                'font_color': '#FFFFFF',
                'border': 1
            })
            
            for col_num, value in enumerate(report_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Set column widths
            worksheet.set_column('A:A', 45) # Product Name
            worksheet.set_column('B:B', 22) # SKU Code
            worksheet.set_column('C:C', 20) # Category
            worksheet.set_column('D:D', 24) # Date Period
            worksheet.set_column('E:E', 18) # Predicted Value
            worksheet.set_column('F:F', 16) # Lower Bound
            worksheet.set_column('G:G', 16) # Upper Bound

    print(f"Successfully generated Excel report for 588 SKUs ({len(report_df)} rows)!")
    print(f"Saved to: {out_excel_1}")
    print(f"Total Predicted Units (1 Aug - 11 Aug) : {report_df['Predicted Value'].sum():,} units")
    print(f"Total Upper Safety Buffer             : {report_df['Upper Bound'].sum():,} units")

if __name__ == '__main__':
    generate_report()
