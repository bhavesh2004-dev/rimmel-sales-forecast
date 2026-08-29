import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.multi_sku_forecaster import run_top_20_pipeline

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..')
    excel_path = os.path.join(project_dir, 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
    output_dir = os.path.join(base_dir, 'output_results')
    
    print("Starting Version 3.0 Top 20 SKUs Pipeline (Price Elasticity & Category Features from SQL/Excel)...\n")
    master_summary_df, master_august_df, cleaned_dict = run_top_20_pipeline(excel_path, output_dir)
    
    print("\nVERSION 3.0 TOP 20 SKUs SUMMARY:")
    print("="*85)
    print(master_summary_df[['SKU', 'Category', 'Avg Price ($ USD)', 'Model Assigned', 'July Vol Accuracy %', 'August Total Forecast']].to_string(index=False))

if __name__ == '__main__':
    main()
