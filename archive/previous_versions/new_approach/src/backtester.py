"""
Backtester Module
Evaluates the forecaster across multiple 30-day historical test windows and plots actuals vs forecasts.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

from .model import SalesForecasterEnsemble

def run_backtesting(clean_data: pd.DataFrame, feature_cols: list, windows: list, output_dir: str, sku_code: str):
    results = []
    fig, axes = plt.subplots(len(windows), 1, figsize=(15, 4 * len(windows)), dpi=300, sharex=False)
    if len(windows) == 1:
        axes = [axes]
        
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    feature_importance_list = []
    
    for idx, win in enumerate(windows):
        test_start = pd.to_datetime(win['test_start'])
        test_end = pd.to_datetime(win['test_end'])
        
        train_df = clean_data[clean_data['date'] < test_start].copy()
        test_df = clean_data[(clean_data['date'] >= test_start) & (clean_data['date'] <= test_end)].copy()
        
        X_train, y_train = train_df[feature_cols], train_df['target_sales']
        X_test, y_test = test_df[feature_cols], test_df['sold_qty']
        
        forecaster = SalesForecasterEnsemble()
        forecaster.fit(X_train, y_train)
        
        pred_train = forecaster.predict(X_train)
        pred_test = forecaster.predict(X_test)
        
        residuals = y_train - pred_train
        res_std = np.std(residuals)
        
        mae = mean_absolute_error(y_test, pred_test)
        total_actual = y_test.sum()
        total_pred = pred_test.sum()
        
        mape = np.mean(np.abs((y_test - pred_test) / np.maximum(y_test, 1))) * 100
        monthly_acc = (1 - abs(total_actual - total_pred) / total_actual) * 100
        
        results.append({
            'Window': win['name'],
            'MAE': mae,
            'Actual Monthly Total': total_actual,
            'Predicted Monthly Total': total_pred,
            'Monthly Volume Accuracy %': monthly_acc,
            'Daily MAPE %': mape
        })
        
        feature_importance_list.append(forecaster.get_feature_importances())
        
        # Plotting
        ax = axes[idx]
        plot_start = test_start - pd.Timedelta(days=60)
        context_df = clean_data[(clean_data['date'] >= plot_start) & (clean_data['date'] <= test_end)]
        
        ax.plot(context_df['date'], context_df['sold_qty'], label='Historical Actual Sales', color='#2c3e50', alpha=0.6, linewidth=1.5)
        test_dates = test_df['date']
        ax.plot(test_dates, pred_test, label='30-Day Model Forecast', color='#e67e22', linewidth=2.5, marker='o', markersize=4)
        
        lower_bound = np.maximum(0, pred_test - 1.96 * res_std)
        upper_bound = pred_test + 1.96 * res_std
        ax.fill_between(test_dates, lower_bound, upper_bound, color='#f39c12', alpha=0.25, label='95% Residual Uncertainty Band')
        
        ax.set_title(f"{win['name']} | MAE: {mae:.2f} | Monthly Total Accuracy: {monthly_acc:.1f}% (Actual: {int(total_actual)}, Pred: {int(total_pred)})", fontsize=12, fontweight='bold')
        ax.set_ylabel('Daily Units Sold', fontsize=10)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.suptitle(f'Pilot SKU ({sku_code}) 30-Day Direct Multi-Window Backtesting Forecasts', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'pilot_sku_backtest_forecasts.png')
    plt.savefig(chart_path)
    plt.close()
    
    avg_fi = np.mean(feature_importance_list, axis=0)
    fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': avg_fi}).sort_values('Importance', ascending=False)
    res_df = pd.DataFrame(results)
    
    res_df.to_csv(os.path.join(output_dir, 'backtest_summary_metrics.csv'), index=False)
    fi_df.to_csv(os.path.join(output_dir, 'feature_importance_summary.csv'), index=False)
    
    return res_df, fi_df
