"""
Rimmel Sales Forecasting & Dynamic Inventory Decision Platform
===============================================================
Production Edition (Modular Architecture v8.1 - Enhanced Visual Analytics)

Features:
1. Mode 1: Dynamic Production Forecast - User-selectable future horizon (up to 31 days).
2. Mode 2: Protected Internal Holdout Validation (21–31 July 2026) - Fixed benchmark gate.
3. Visual Product Inspector: Clear historical sales line + 3 distinct forecast reference lines
   (Baseline, Momentum, Recommended) across the designated forecast window.
4. Data-Grounded Decision Explanations (v7d, v14d, v30d, v90d, CV, and stock signals).

Run: streamlit run app.py
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add project root to path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.settings import (
    DB_PATH, HOLDOUT_TRAIN_START, HOLDOUT_TRAIN_END,
    HOLDOUT_EVAL_START, HOLDOUT_EVAL_END, HOLDOUT_DAYS,
    DEFAULT_PROD_TRAIN_START, DEFAULT_PROD_CUTOFF,
    MAX_FORECAST_HORIZON_DAYS, REPORTS_DIR
)
from src.data_loader import load_sales_data
from src.dynamic_engine import run_dynamic_forecast
from src.validation import run_holdout_benchmark, evaluate_holdout_performance
from src.report_generator import generate_client_excel_report

st.set_page_config(
    page_title="Rimmel Forecasting & Inventory Decision Platform",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Theme Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.metric-box-green {
    background: linear-gradient(135deg, #1f6f3e 0%, #114b27 100%);
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffffff;
    box-shadow: 0 3px 10px rgba(35, 134, 54, 0.3);
}
.metric-box-blue {
    background: linear-gradient(135deg, #1f6feb 0%, #0d419d 100%);
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffffff;
    box-shadow: 0 3px 10px rgba(31, 111, 235, 0.3);
}
.metric-box-purple {
    background: linear-gradient(135deg, #6e40c9 0%, #4c2889 100%);
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffffff;
    box-shadow: 0 3px 10px rgba(110, 64, 201, 0.3);
}
.metric-box-red {
    background: linear-gradient(135deg, #b71c1c 0%, #7f0000 100%);
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffffff;
    box-shadow: 0 3px 10px rgba(183, 28, 28, 0.3);
}
.protocol-ribbon {
    background: #161b22;
    border-left: 4px solid #238636;
    border-radius: 4px;
    padding: 10px 16px;
    margin-bottom: 14px;
    font-size: 0.88rem;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_master_data():
    return load_sales_data(DB_PATH)

df_master_sales = get_master_data()
total_catalog_skus = df_master_sales['sku'].nunique()

# ── Sidebar: Mode Selection & Controls ───────────────────────────────────────────
st.sidebar.title("💄 Navigation & Controls")

app_mode = st.sidebar.radio(
    "🎯 Select System Mode",
    [
        "🚀 Mode 1: Dynamic Production Forecast",
        "🧪 Mode 2: Protected Internal Holdout Validation (21–31 July 2026)"
    ],
    index=0
)

st.sidebar.markdown("---")

def render_forecast_graph(df_sku_history, p_baseline, p_momentum, p_recommended, 
                          forecast_start_ts, forecast_end_ts, horizon_days, sku_name, 
                          actual_holdout_series=None, history_window="60 Days"):
    """
    Renders an uncluttered, client-friendly graph with historical sales line 
    and three visually distinct forecast reference lines across the forecast horizon.
    """
    cutoff_date = forecast_start_ts - pd.Timedelta(days=1)
    
    # Filter historical window
    if history_window == "30 Days":
        start_hist = cutoff_date - pd.Timedelta(days=30)
    elif history_window == "60 Days":
        start_hist = cutoff_date - pd.Timedelta(days=60)
    elif history_window == "90 Days":
        start_hist = cutoff_date - pd.Timedelta(days=90)
    else:
        start_hist = df_sku_history['date'].min()
        
    df_plot_hist = df_sku_history[(df_sku_history['date'] >= start_hist) & (df_sku_history['date'] <= cutoff_date)].copy()
    
    # Daily forecast rates
    daily_base = p_baseline / float(horizon_days)
    daily_mom  = p_momentum / float(horizon_days)
    daily_rec  = p_recommended / float(horizon_days)
    
    forecast_dates = pd.date_range(start=forecast_start_ts, end=forecast_end_ts)
    
    fig = go.Figure()
    
    # 1. Historical Actual Daily Sales Line
    fig.add_trace(go.Scatter(
        x=df_plot_hist['date'], y=df_plot_hist['total_sales'],
        mode='lines+markers', name='Actual Historical Sales (Daily)',
        line=dict(color='#58a6ff', width=2),
        marker=dict(size=5, color='#58a6ff')
    ))
    
    # 2. Historical 7-Day Moving Average Trend
    df_plot_hist['ma7'] = df_plot_hist['total_sales'].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_plot_hist['date'], y=df_plot_hist['ma7'],
        mode='lines', name='7-Day Historical Trend',
        line=dict(color='#8b949e', width=1.5, dash='dot')
    ))
    
    # 3. If in Holdout Mode, plot ground-truth actuals in the holdout period
    if actual_holdout_series is not None and len(actual_holdout_series) > 0:
        fig.add_trace(go.Scatter(
            x=actual_holdout_series['date'], y=actual_holdout_series['total_sales'],
            mode='lines+markers', name='Actual Holdout Sales (Ground Truth)',
            line=dict(color='#39d353', width=2.5),
            marker=dict(size=6, symbol='diamond', color='#39d353')
        ))
        
    # 4. Three Distinct Forecast Reference Lines across the forecast window
    # 🛡️ Baseline Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=[daily_base] * len(forecast_dates),
        mode='lines+markers', name=f'🛡️ Baseline Anchor ({p_baseline:,} u total | {daily_base:.1f} u/d)',
        line=dict(color='#d29922', width=2.5, dash='dash'),
        marker=dict(size=4, color='#d29922')
    ))
    
    # ⚡ Momentum Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=[daily_mom] * len(forecast_dates),
        mode='lines+markers', name=f'⚡ Momentum Run-Rate ({p_momentum:,} u total | {daily_mom:.1f} u/d)',
        line=dict(color='#a371f7', width=2.5, dash='dashdot'),
        marker=dict(size=4, color='#a371f7')
    ))
    
    # 🎯 Recommended / Adaptive Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=[daily_rec] * len(forecast_dates),
        mode='lines+markers', name=f'🎯 Recommended Forecast ({p_recommended:,} u total | {daily_rec:.1f} u/d)',
        line=dict(color='#2ea043', width=3.5),
        marker=dict(size=6, color='#2ea043')
    ))
    
    # 5. Shaded Forecast Region & Vertical Boundary
    fig.add_vrect(
        x0=forecast_start_ts - pd.Timedelta(hours=12),
        x1=forecast_end_ts + pd.Timedelta(hours=12),
        fillcolor="rgba(35, 134, 54, 0.08)", opacity=1,
        layer="below", line_width=1, line_dash="dash", line_color="#2ea043"
    )
    
    fig.update_layout(
        title=f"<b>Historical Demand vs. Three Forecasting Approaches:</b> {sku_name}",
        template='plotly_dark',
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        height=450,
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis=dict(title="Date", showgrid=True, gridcolor='#21262d'),
        yaxis=dict(title="Daily Demand (Units / Day)", showgrid=True, gridcolor='#21262d'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(22, 27, 34, 0.85)',
            bordercolor='#30363d',
            borderwidth=1
        ),
        hovermode="x unified"
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════════
# MODE 1: DYNAMIC PRODUCTION FORECAST (USER-SELECTABLE HORIZON)
# ═══════════════════════════════════════════════════════════════════════════════════
if app_mode == "🚀 Mode 1: Dynamic Production Forecast":
    st.title("💄 Rimmel Production Forecasting & Inventory Decision System")
    
    st.markdown("""
    <div class="protocol-ribbon">
        🚀 <b>Dynamic Production Forecast:</b> Select any forward date horizon up to 31 days. The model automatically trains on all historical data up to 31 July 2026 and dynamically scales daily demand rates.
    </div>
    """, unsafe_allow_html=True)
    
    # Date Range Selector
    c_date1, c_date2, c_date3 = st.columns([1.5, 1.5, 2.0])
    
    min_allowed_date = pd.to_datetime('2026-08-01').date()
    max_allowed_date = pd.to_datetime('2026-12-31').date()
    
    forecast_start_date = c_date1.date_input(
        "📅 Forecast Start Date",
        value=pd.to_datetime('2026-08-01').date(),
        min_value=min_allowed_date,
        max_value=max_allowed_date
    )
    
    forecast_end_date = c_date2.date_input(
        "📅 Forecast End Date",
        value=pd.to_datetime('2026-08-11').date(),
        min_value=forecast_start_date,
        max_value=max_allowed_date
    )
    
    # Calculate Horizon
    selected_horizon_days = (forecast_end_date - forecast_start_date).days + 1
    
    if selected_horizon_days > MAX_FORECAST_HORIZON_DAYS:
        st.warning(f"⚠️ Selected horizon is {selected_horizon_days} days. Recommended maximum is {MAX_FORECAST_HORIZON_DAYS} days (1 month).")
        
    c_date3.metric(
        "⏱️ Forecast Horizon",
        f"{selected_horizon_days} Days",
        delta=f"{forecast_start_date.strftime('%d/%m/%Y')} → {forecast_end_date.strftime('%d/%m/%Y')}"
    )
    
    # Run Dynamic Forecast
    @st.cache_data(ttl=60)
    def get_dynamic_prod_forecast(start_d, end_d):
        return run_dynamic_forecast(
            df_master_sales,
            forecast_start=start_d,
            forecast_end=end_d
        )
        
    df_prod_forecast = get_dynamic_prod_forecast(forecast_start_date, forecast_end_date)
    
    # Key Macro Metrics
    total_forecast_units = int(df_prod_forecast['Recommended Forecast'].sum())
    dead_stock_count     = len(df_prod_forecast[df_prod_forecast['Inventory Health Status'] == 'DEAD / STUCK'])
    slow_moving_count    = len(df_prod_forecast[df_prod_forecast['Inventory Health Status'] == 'SLOW MOVING'])
    
    m_col1, m_col2, m_col3, m_col4 = st.columns([1.2, 1.4, 1.4, 1.8])
    m_col1.markdown(f"""
    <div class="metric-box-green">
        <div style="font-size: 0.8rem; opacity: 0.85;">🏆 Master Catalog</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{total_catalog_skus} SKUs</div>
        <div style="font-size: 0.75rem; opacity: 0.85;">100% Real-Data Grounded</div>
    </div>
    """, unsafe_allow_html=True)
    
    m_col2.markdown(f"""
    <div class="metric-box-blue">
        <div style="font-size: 0.8rem; opacity: 0.85;">🎯 Total Planned Demand</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{total_forecast_units:,} <span style="font-size: 0.85rem; font-weight: 400;">units</span></div>
        <div style="font-size: 0.75rem; opacity: 0.85;">{selected_horizon_days}-Day Scaled Forecast</div>
    </div>
    """, unsafe_allow_html=True)
    
    m_col3.markdown(f"""
    <div class="metric-box-red">
        <div style="font-size: 0.8rem; opacity: 0.85;">⚠️ Potential Dead / Slow Stock</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{dead_stock_count + slow_moving_count} <span style="font-size: 0.85rem; font-weight: 400;">SKUs</span></div>
        <div style="font-size: 0.75rem; opacity: 0.85;">{dead_stock_count} Dead + {slow_moving_count} Slow Moving</div>
    </div>
    """, unsafe_allow_html=True)
    
    with m_col4:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        # Generate Dynamic Excel on the fly
        report_filename = f"Rimmel_Forecast_{forecast_start_date.strftime('%Y%m%d')}_to_{forecast_end_date.strftime('%Y%m%d')}.xlsx"
        report_filepath = os.path.join(REPORTS_DIR, report_filename)
        saved_excel_path = generate_client_excel_report(df_prod_forecast, report_filepath, is_evaluation=False)
        
        with open(saved_excel_path, 'rb') as f_excel:
            excel_bytes = f_excel.read()
            
        st.download_button(
            label=f"📥 Download Excel ({selected_horizon_days} Days)",
            data=excel_bytes,
            file_name=report_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    st.markdown("---")
    
    # ── Production Tabs ───────────────────────────────────────────────────────────
    prod_tab1, prod_tab2, prod_tab3 = st.tabs([
        "📋 Executive Forecast & Planning Table (Clean 8 Columns)",
        "📈 Visual Product Inspector (Historical vs. 3 Forecast Lines)",
        "🚨 Inventory Action Summary (Dead & Slow Stock)"
    ])
    
    with prod_tab1:
        st.markdown("### 📋 Executive Forecast & Planning Table")
        st.caption("Answers: 1. How much should I expect to sell? 2. How much can I trust this number? 3. What is the data-backed reason?")
        
        col_f1, col_f2, col_f3 = st.columns([2, 1.2, 1.2])
        search_query = col_f1.text_input("🔍 Search SKU or Category", "", placeholder="e.g. RIM-MSC, RIM-SMPP, Mascara")
        conf_filter  = col_f2.selectbox("Filter by Confidence", ["All Confidence Levels", "HIGH Only", "MEDIUM Only", "LOW Only"])
        status_filter= col_f3.selectbox("Filter by Risk / Status", ["All Statuses", "HIGH DEMAND", "DEMAND INCREASING", "DEMAND DECLINING", "VOLATILE", "STOCK RISK", "LOW DEMAND", "DEAD / NEAR-DEAD"])
        
        client_columns = [
            'Date', 'Product SKU', 'Baseline Prediction', 'Momentum Prediction',
            'Recommended Forecast', 'Confidence', 'Risk / Status', 'Reason'
        ]
        df_display = df_prod_forecast[client_columns].copy()
        
        if search_query:
            df_display = df_display[df_display['Product SKU'].str.contains(search_query, case=False)]
            
        if conf_filter == "HIGH Only":
            df_display = df_display[df_display['Confidence'] == 'HIGH']
        elif conf_filter == "MEDIUM Only":
            df_display = df_display[df_display['Confidence'] == 'MEDIUM']
        elif conf_filter == "LOW Only":
            df_display = df_display[df_display['Confidence'] == 'LOW']
            
        if status_filter != "All Statuses":
            df_display = df_display[df_display['Risk / Status'] == status_filter]
            
        st.dataframe(
            df_display.style.format({
                'Baseline Prediction': '{:,.0f}',
                'Momentum Prediction': '{:,.0f}',
                'Recommended Forecast': '{:,.0f}'
            }),
            use_container_width=True,
            height=520
        )
        
    with prod_tab2:
        st.markdown("### 📈 Visual Product Inspector: Demand History & Forecast Lines")
        st.caption("Shows what the product sold historically and what the three forecasting approaches expect over the selected horizon.")
        
        sku_rank = df_master_sales.groupby('sku')['total_sales'].sum().sort_values(ascending=False).index.tolist()
        annual_map = df_master_sales.groupby('sku')['total_sales'].sum().to_dict()
        sku_options = [f"{sku}  —  ({int(annual_map.get(sku, 0)):,} u/yr)" for sku in sku_rank]
        
        c_sel1, c_sel2 = st.columns([3, 1])
        selected_display = c_sel1.selectbox("📦 Select Product SKU to Inspect", sku_options, index=0)
        selected_sku = selected_display.split("  —  ")[0].strip()
        hist_window = c_sel2.selectbox("History Window", ["60 Days", "30 Days", "90 Days", "Full History"], index=0)
        
        df_sku_hist = df_master_sales[df_master_sales['sku'] == selected_sku].sort_values('date')
        sku_row = df_prod_forecast[df_prod_forecast['Product SKU'] == selected_sku]
        
        if not sku_row.empty:
            p_b = int(sku_row['Baseline Prediction'].values[0])
            p_m = int(sku_row['Momentum Prediction'].values[0])
            p_r = int(sku_row['Recommended Forecast'].values[0])
            c_l = sku_row['Confidence'].values[0]
            s_l = sku_row['Risk / Status'].values[0]
            r_l = sku_row['Reason'].values[0]
            d_i = sku_row['Estimated Days of Inventory'].values[0]
            h_s = sku_row['Inventory Health Status'].values[0]
            a_r = sku_row['Inventory Action Recommendation'].values[0]
            stk = int(sku_row['Current Stock (Units)'].values[0])
        else:
            p_b, p_m, p_r, c_l, s_l, r_l, d_i, h_s, a_r, stk = 0, 0, 0, "LOW", "NORMAL", "N/A", "N/A", "HEALTHY", "None", 0
            
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🛡️ Baseline Organic", f"{p_b:,} u", delta=f"{p_b/selected_horizon_days:.1f} u/d")
        m2.metric("⚡ Momentum Trend", f"{p_m:,} u", delta=f"{p_m/selected_horizon_days:.1f} u/d")
        m3.metric("🎯 Recommended Forecast", f"{p_r:,} u", delta=f"{p_r/selected_horizon_days:.1f} u/d")
        m4.metric("Confidence / Status", f"{c_l} / {s_l}")
        m5.metric("Current Stock / Days", f"{stk:,} u ({d_i})")
        
        st.info(f"💡 **Data-Grounded Rationale:** {r_l}\n\n📦 **Inventory Recommendation:** {a_r} (Health Status: **{h_s}**)")
        
        fig_prod = render_forecast_graph(
            df_sku_hist, p_b, p_m, p_r,
            pd.to_datetime(forecast_start_date),
            pd.to_datetime(forecast_end_date),
            selected_horizon_days,
            selected_sku,
            history_window=hist_window
        )
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with prod_tab3:
        st.markdown("### 🚨 Inventory Action Summary (Dead / Stuck Inventory & Overstock)")
        st.caption("Identifies slow-moving or structurally stuck products to help procurement recover trapped working capital.")
        
        inv_columns = [
            'Product SKU', 'Category', 'Current Stock (Units)', 'Selling Price ($)',
            'Recommended Daily Demand', 'Estimated Days of Inventory',
            'Inventory Health Status', 'Inventory Action Recommendation', 'Annual Sales (Units)'
        ]
        df_inv_tab = df_prod_forecast[inv_columns].copy()
        df_inv_tab['Trapped Working Capital ($)'] = round(df_inv_tab['Current Stock (Units)'] * df_inv_tab['Selling Price ($)'], 2)
        
        inv_filter = st.selectbox("Filter Inventory Health", ["All Inventory", "DEAD / STUCK Only", "SLOW MOVING Only", "WATCH Only", "HEALTHY Only"])
        if inv_filter != "All Inventory":
            target_status = inv_filter.replace(" Only", "")
            df_inv_tab = df_inv_tab[df_inv_tab['Inventory Health Status'] == target_status]
            
        st.dataframe(
            df_inv_tab.style.format({
                'Current Stock (Units)': '{:,.0f}',
                'Selling Price ($)': '${:.2f}',
                'Recommended Daily Demand': '{:.2f}',
                'Trapped Working Capital ($)': '${:,.2f}',
                'Annual Sales (Units)': '{:,.0f}'
            }),
            use_container_width=True,
            height=500
        )

# ═══════════════════════════════════════════════════════════════════════════════════
# MODE 2: PROTECTED INTERNAL HOLDOUT VALIDATION (21–31 JULY 2026)
# ═══════════════════════════════════════════════════════════════════════════════════
else:
    st.title("🧪 Protected Internal Holdout Validation Gate (21–31 July 2026)")
    
    st.markdown("""
    <div class="protocol-ribbon">
        🔒 <b>Protected Evaluation Gate:</b> Trained strictly on data <code>1 Aug 2025 → 20 Jul 2026</code>. Evaluated against ground-truth actuals from <code>21 Jul 2026 → 31 Jul 2026</code> with <b>zero data leakage</b>.
    </div>
    """, unsafe_allow_html=True)
    
    # Run Holdout Simulation (Isolated in src/validation.py)
    @st.cache_data(ttl=60)
    def run_holdout_evaluation():
        return run_holdout_benchmark(df_master_sales)
        
    df_holdout_evaluated, holdout_metrics = run_holdout_evaluation()
    
    # Metrics Row
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Ground Truth Actual Sales", f"{holdout_metrics['total_actual_units']:,} units")
    h2.metric("Baseline Model WAPE", f"{holdout_metrics['wape_baseline_pct']:.2f}%")
    h3.metric("Momentum Model WAPE", f"{holdout_metrics['wape_momentum_pct']:.2f}%")
    h4.metric("Recommended Forecast WAPE", f"{holdout_metrics['wape_recommended_pct']:.2f}%", delta="🏆 Final Production Model")
    
    st.markdown("---")
    
    # Holdout Tabs
    h_tab1, h_tab2, h_tab3 = st.tabs([
        "📊 SKU-Level Holdout Evaluation Table (Actuals vs Predictions)",
        "📈 Visual Holdout Product Inspector (Actuals vs 3 Forecast Lines)",
        "⚖️ Head-to-Head Win Breakdown & Calibration"
    ])
    
    with h_tab1:
        st.markdown("### 📊 SKU-Level Holdout Evaluation (Actuals vs Predictions)")
        
        holdout_cols = [
            'Date', 'Product SKU', 'Category', 'Actual Sales',
            'Baseline Prediction', 'Momentum Prediction', 'Recommended Forecast',
            'Recommended Error', 'Model Performance Comparison',
            'Confidence', 'Risk / Status', 'Reason'
        ]
        
        st.dataframe(
            df_holdout_evaluated[holdout_cols].style.format({
                'Actual Sales': '{:,.0f}',
                'Baseline Prediction': '{:,.0f}',
                'Momentum Prediction': '{:,.0f}',
                'Recommended Forecast': '{:,.0f}',
                'Recommended Error': '{:,.0f}'
            }),
            use_container_width=True,
            height=500
        )
        
    with h_tab2:
        st.markdown("### 📈 Visual Holdout Product Inspector (Ground Truth vs 3 Model Forecasts)")
        
        sku_rank_h = df_holdout_evaluated.sort_values('Actual Sales', ascending=False)['Product SKU'].tolist()
        sku_options_h = [f"{sku}  —  (Holdout Actual: {int(df_holdout_evaluated[df_holdout_evaluated['Product SKU']==sku]['Actual Sales'].values[0]):,} u)" for sku in sku_rank_h]
        
        c_hsel1, c_hsel2 = st.columns([3, 1])
        selected_display_h = c_hsel1.selectbox("📦 Select SKU to Inspect on Holdout", sku_options_h, index=0)
        selected_sku_h = selected_display_h.split("  —  ")[0].strip()
        hist_window_h = c_hsel2.selectbox("History Window (Holdout)", ["60 Days", "30 Days", "90 Days", "Full History"], index=0)
        
        sku_h_row = df_holdout_evaluated[df_holdout_evaluated['Product SKU'] == selected_sku_h]
        df_sku_hist_h = df_master_sales[df_master_sales['sku'] == selected_sku_h].sort_values('date')
        
        act_val = int(sku_h_row['Actual Sales'].values[0])
        p_b_h   = int(sku_h_row['Baseline Prediction'].values[0])
        p_m_h   = int(sku_h_row['Momentum Prediction'].values[0])
        p_r_h   = int(sku_h_row['Recommended Forecast'].values[0])
        err_h   = int(sku_h_row['Recommended Error'].values[0])
        comp_h  = sku_h_row['Model Performance Comparison'].values[0]
        r_l_h   = sku_h_row['Reason'].values[0]
        
        m_h1, m_h2, m_h3, m_h4, m_h5 = st.columns(5)
        m_h1.metric("🎯 Ground Truth Actual", f"{act_val:,} u")
        m_h2.metric("🛡️ Baseline Forecast", f"{p_b_h:,} u", delta=f"Error: {abs(act_val-p_b_h):,} u")
        m_h3.metric("⚡ Momentum Forecast", f"{p_m_h:,} u", delta=f"Error: {abs(act_val-p_m_h):,} u")
        m_h4.metric("🏆 Recommended Forecast", f"{p_r_h:,} u", delta=f"Error: {err_h:,} u")
        m_h5.metric("Winner Outcome", comp_h)
        
        st.info(f"💡 **Data-Grounded Rationale:** {r_l_h}")
        
        # Extract actual holdout daily slice
        df_holdout_actual_slice = df_sku_hist_h[
            (df_sku_hist_h['date'] >= pd.to_datetime(HOLDOUT_EVAL_START)) &
            (df_sku_hist_h['date'] <= pd.to_datetime(HOLDOUT_EVAL_END))
        ]
        
        fig_holdout = render_forecast_graph(
            df_sku_hist_h, p_b_h, p_m_h, p_r_h,
            pd.to_datetime(HOLDOUT_EVAL_START),
            pd.to_datetime(HOLDOUT_EVAL_END),
            HOLDOUT_DAYS,
            selected_sku_h,
            actual_holdout_series=df_holdout_actual_slice,
            history_window=hist_window_h
        )
        st.plotly_chart(fig_holdout, use_container_width=True)
        
    with h_tab3:
        st.markdown("### ⚖️ Head-to-Head Win Breakdown & Calibration")
        
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            st.markdown(f"""
            <div style="background: #161b22; padding: 16px; border-radius: 8px;">
                <h4>🏆 Model Win Distribution (588 SKUs)</h4>
                <ul>
                    <li><b>Momentum Performed Better:</b> {holdout_metrics['momentum_winner_count']} SKUs ({holdout_metrics['momentum_winner_count']/total_catalog_skus*100:.1f}%)</li>
                    <li><b>Baseline Performed Better:</b> {holdout_metrics['baseline_winner_count']} SKUs ({holdout_metrics['baseline_winner_count']/total_catalog_skus*100:.1f}%)</li>
                    <li><b>Tied / Approximately Equal:</b> {holdout_metrics['tied_winner_count']} SKUs ({holdout_metrics['tied_winner_count']/total_catalog_skus*100:.1f}%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with c_w2:
            conf_calib = df_holdout_evaluated.groupby('Confidence').agg(
                SKU_Count=('Product SKU', 'count'),
                Total_Actual=('Actual Sales', 'sum'),
                Recommended_Error=('Recommended Error', 'sum')
            ).reset_index()
            conf_calib['WAPE (%)'] = round(conf_calib['Recommended_Error'] / conf_calib['Total_Actual'] * 100, 2)
            
            st.markdown("#### 🎯 Evidence-Based Confidence Calibration")
            st.dataframe(conf_calib, use_container_width=True)
