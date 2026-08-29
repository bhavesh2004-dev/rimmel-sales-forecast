import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import io

# ── Page Configuration ──
st.set_page_config(
    page_title="Rimmel Sales Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 24px; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
    .sub-header { font-size: 13px; color: #64748b; margin-bottom: 20px; }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .kpi-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
    .kpi-value { font-size: 22px; font-weight: 700; color: #0f172a; }
    .kpi-sub { font-size: 11px; color: #10b981; font-weight: 500; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Database Path ──
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH  = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')

@st.cache_data
def load_data_from_sqlite():
    if not os.path.exists(DB_PATH):
        return None, None, None
        
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load History
    df_hist = pd.read_sql("SELECT * FROM full_history", conn)
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist['sku']  = df_hist['sku'].astype(str).str.strip()
    
    # 2. Load 3-Tier Master Production Forecast
    try:
        df_fore = pd.read_sql("SELECT * FROM forecast_aug_2026_3tier_master", conn)
    except Exception:
        df_fore = pd.read_sql("SELECT * FROM forecast_aug_2026", conn)
            
    conn.close()
    
    # Ensure exact 8-column naming
    column_mapping = {
        'order sku': 'product',
        'predicted unit': 'predicted value',
        'lower bound': 'lower bound',
        'upper bound': 'upper bound',
        'tier_engine': 'tier category'
    }
    df_fore = df_fore.rename(columns=column_mapping)
    df_fore['product'] = df_fore['product'].astype(str).str.strip()
    
    # 3. Create SKU Summary
    df_smry = df_hist.groupby('sku').agg(
        total_units = ('total_sales', 'sum'),
        avg_price   = ('selling_price', 'median'),
        category    = ('category', 'first')
    ).reset_index()
    
    return df_hist, df_fore, df_smry

def generate_exact_excel_report(df_sub):
    column_order = [
        'product',
        'dates',
        'predicted value',
        'actual value',
        'percentage error',
        'lower bound',
        'upper bound',
        'tier category'
    ]
    report_df = df_sub[column_order].sort_values(by='predicted value', ascending=False)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        report_df.to_excel(writer, sheet_name='Sheet1', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
        for col_num, value in enumerate(report_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:A', 26)
        worksheet.set_column('B:B', 28)
        worksheet.set_column('C:G', 18)
        worksheet.set_column('H:H', 45)
    return buffer.getvalue()

df_history, df_forecasts, df_summary = load_data_from_sqlite()

st.markdown('<div class="main-header">📈 Rimmel Sales Forecaster & Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by 3-Tier Master Production Engine (Aug 2025 – Jul 2026 Data)</div>', unsafe_allow_html=True)

if df_forecasts is not None and df_history is not None:

    # ── Sidebar Controls ──
    st.sidebar.markdown("### 🎛️ Controls")
    
    all_skus_sorted = df_forecasts.sort_values('predicted value', ascending=False)['product'].tolist()

    catalog_scope = st.sidebar.radio("Catalog Filter", ["Top 20 SKUs", "All 514 SKUs"], index=0)
    
    if catalog_scope == "Top 20 SKUs":
        active_sku_options = all_skus_sorted[:20]
    else:
        active_sku_options = all_skus_sorted

    selected_sku = st.sidebar.selectbox("Select Product SKU", options=active_sku_options, index=0)

    # Filter for selected SKU
    sku_forecast = df_forecasts[df_forecasts['product'] == selected_sku].iloc[0]
    
    sku_hist = df_history[df_history['sku'] == selected_sku].sort_values('date')
    cat_name = sku_hist['category'].iloc[0] if len(sku_hist) > 0 else 'Cosmetics'
    tier_engine_name = sku_forecast.get('tier category', '3-Tier Master Engine')

    pred_units  = int(sku_forecast['predicted value'])
    act_units   = int(sku_forecast['actual value'])
    pct_err_str = str(sku_forecast['percentage error'])
    upper_units = int(sku_forecast['upper bound'])
    lower_units = int(sku_forecast['lower bound'])

    # Direct Excel Export
    excel_bytes_sidebar = generate_exact_excel_report(df_forecasts)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 1-Click Direct Export")
    st.sidebar.download_button(
        label=f"📊 Download Forecast Excel Report",
        data=excel_bytes_sidebar,
        file_name=f"Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

    # ── Top KPI Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Product SKU</div>
            <div class="kpi-value" style="font-size: 15px;">{selected_sku}</div>
            <div class="kpi-sub" style="color: #64748b;">Category: {cat_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">August 1–11 Forecast</div>
            <div class="kpi-value">{pred_units:,} <span style="font-size: 13px; font-weight: normal;">units</span></div>
            <div class="kpi-sub">Actual: {act_units:,} units ({pct_err_str})</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Upper Safety Buffer</div>
            <div class="kpi-value">{upper_units:,} <span style="font-size: 13px; font-weight: normal;">units</span></div>
            <div class="kpi-sub" style="color: #0284c7;">Upper Confidence Limit</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lower Confidence Bound</div>
            <div class="kpi-value">{lower_units:,} <span style="font-size: 13px; font-weight: normal;">units</span></div>
            <div class="kpi-sub" style="color: #64748b;">Conservative Estimate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📊 Historical & August Forecast Chart", "📋 Exact Forecast Table", "📥 Export Reports"])

    with tab1:
        st.markdown(f"#### Sales Trend & August 1–11, 2026 Forecast (`{tier_engine_name}`)")
        
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=200)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        # 1. Historical Sales (Last 60 Days: June 1 to July 31, 2026)
        max_h_date = pd.to_datetime('2026-07-31')
        min_h_date = max_h_date - pd.Timedelta(days=60)
        
        sku_hist_sub = sku_hist[(sku_hist['date'] >= min_h_date) & (sku_hist['date'] <= max_h_date)]
        
        if len(sku_hist_sub) > 0:
            ax.plot(sku_hist_sub['date'], sku_hist_sub['total_sales'], label='Historical Sales (June - July 2026)', color='#3b82f6', linewidth=2.0, marker='o', markersize=3)
            
        # 2. August 1 to August 11 Forecast Line & 95% Confidence Band
        aug_dates = pd.date_range('2026-08-01', '2026-08-11', freq='D')
        daily_forecast_val = pred_units / 11.0
        daily_upper_val    = upper_units / 11.0
        daily_lower_val    = lower_units / 11.0
        
        forecast_line = np.full(len(aug_dates), daily_forecast_val)
        upper_line    = np.full(len(aug_dates), daily_upper_val)
        lower_line    = np.full(len(aug_dates), daily_lower_val)
        
        # Connect July 31 to August 1
        if len(sku_hist_sub) > 0:
            last_hist_date = sku_hist_sub['date'].iloc[-1]
            last_hist_val  = sku_hist_sub['total_sales'].iloc[-1]
            
            conn_dates = [last_hist_date] + list(aug_dates)
            conn_fcst  = [last_hist_val] + list(forecast_line)
            conn_upper = [last_hist_val] + list(upper_line)
            conn_lower = [last_hist_val] + list(lower_line)
        else:
            conn_dates = list(aug_dates)
            conn_fcst  = list(forecast_line)
            conn_upper = list(upper_line)
            conn_lower = list(lower_line)

        ax.plot(conn_dates[1:], conn_fcst[1:], label='August 1-11 Forecast (3-Tier Master)', color='#10b981', linewidth=2.5, linestyle='--')
        ax.fill_between(conn_dates[1:], conn_lower[1:], conn_upper[1:], color='#10b981', alpha=0.18, label='95% Confidence Buffer')

        ax.set_ylabel("Daily Units Sold", fontsize=10, fontweight='bold', color='#334155')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9)
        
        st.pyplot(fig)

    with tab2:
        exact_columns = [
            'product',
            'dates',
            'predicted value',
            'actual value',
            'percentage error',
            'lower bound',
            'upper bound',
            'tier category'
        ]
        st.dataframe(
            df_forecasts[exact_columns],
            use_container_width=True,
            height=420
        )

    with tab3:
        st.markdown("#### Download Production Reports")
        d1, d2 = st.columns(2)
        
        with d1:
            st.download_button(
                label=f"📊 Download Exact Forecast Excel Report (.xlsx)",
                data=excel_bytes_sidebar,
                file_name=f"Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
        with d2:
            forecast_bytes = df_forecasts[exact_columns].to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📄 Download Exact Forecast CSV",
                data=forecast_bytes,
                file_name=f"rimmel_3tier_forecast_1Aug_to_11Aug_2026.csv",
                mime="text/csv"
            )

else:
    st.warning("Please ensure SQLite database exists at data/rimmel_clean.db.")
