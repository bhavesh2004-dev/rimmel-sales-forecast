"""
Generate Forecast Excel Report matching user's image specification exactly.
Exact Column Headers:
- order sku
- date
- predicted unit
- lower bound
- upper bound

Exact Date Format:
- "1/08/2026 to 11/08/2026"
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
    
    # Format exact requested columns matching image:
    # order sku | date | predicted unit | lower bound | upper bound
    report_df = pd.DataFrame({
        'order sku': agg_df['order_item_sku'],
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': np.round(agg_df['predicted_sales'], 0).astype(int),
        'lower bound': np.round(agg_df['lower_bound_95'], 0).astype(int),
        'upper bound': np.round(agg_df['upper_bound_95'], 0).astype(int)
    })
    
    # Sort by predicted unit descending
    report_df = report_df.sort_values(by='predicted unit', ascending=False).reset_index(drop=True)
    
    out_excel_1 = os.path.join(project_dir, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
    out_excel_2 = os.path.join(base_dir, 'output_results', 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
    
    for out_path in [out_excel_1, out_excel_2]:
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, sheet_name='Sheet1', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            # Simple clean styling matching standard Excel
            header_format = workbook.add_format({
                'bold': True,
                'border': 1
            })
            
            for col_num, value in enumerate(report_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            worksheet.set_column('A:A', 24) # order sku
            worksheet.set_column('B:B', 28) # date
            worksheet.set_column('C:C', 18) # predicted unit
            worksheet.set_column('D:D', 16) # lower bound
            worksheet.set_column('E:E', 16) # upper bound

    print(f"Generated Excel report matching image specification successfully!")
    print(f"Saved to: {out_excel_1}")
    print("\nFirst 5 rows preview:")
    print(report_df.head(5).to_string(index=False))

if __name__ == '__main__':
    generate_report()
