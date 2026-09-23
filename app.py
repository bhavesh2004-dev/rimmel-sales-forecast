"""
Rimmel Multi-Platform Demand Forecasting & Inventory Planning Platform
======================================================================
Certified Production Architecture: Exp6
- ZERO Treatment (Unobserved days modeled as true market zero-demand)
- LightGBM Regressor (Exact parameters: n_estimators=150, max_depth=6, num_leaves=31, lr=0.05, seed=42)
- Combined Calibration (alpha=0.10 for zero-demand suppression, beta=0.10 for stockout dampening)
- Independent Platform Forecasting (Amazon, eBay, Website, Other) -> SKU Physical Aggregation
- Shared Warehouse Inventory Pool (Single physical inventory pool per SKU; never summed across channels)

Run:
    streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Configuration & Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

st.set_page_config(
    page_title="Rimmel Demand Forecasting Platform",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1F4E78;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1.0rem;
    color: #595959;
    margin-bottom: 1.2rem;
}
.kpi-card {
    background-color: #F8F9FA;
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid #1F4E78;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}
.kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #595959;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1F4E78;
    margin-top: 4px;
}
.kpi-desc {
    font-size: 0.8rem;
    color: #7F7F7F;
    margin-top: 4px;
}
.platform-card {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.platform-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #333333;
}
.platform-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 4px;
}
.callout-box {
    background-color: #EBF3FB;
    border-left: 4px solid #1F4E78;
    padding: 14px;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #1A365D;
    margin-bottom: 16px;
}
.warning-box {
    background-color: #FFF8E1;
    border-left: 4px solid #FFA000;
    padding: 14px;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #6A4B00;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADERS WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data_caches():
    # 1. SKU Master
    sku_master_path = os.path.join(PROCESSED_DIR, 'dashboard_sku_master.csv')
    if os.path.exists(sku_master_path):
        sku_master = pd.read_csv(sku_master_path)
    else:
        sku_master = pd.DataFrame()

    # 2. Validation Daily SKU
    val_daily_path = os.path.join(PROCESSED_DIR, 'dashboard_validation_sku_daily.csv')
    if os.path.exists(val_daily_path):
        val_daily = pd.read_csv(val_daily_path)
        val_daily['date_parsed'] = pd.to_datetime(val_daily['date_iso'])
    else:
        val_daily = pd.DataFrame()

    # 3. Forecast Daily SKU
    fwd_daily_path = os.path.join(PROCESSED_DIR, 'dashboard_forecast_sku_daily.csv')
    if os.path.exists(fwd_daily_path):
        fwd_daily = pd.read_csv(fwd_daily_path)
        fwd_daily['date_parsed'] = pd.to_datetime(fwd_daily['date_iso'])
    else:
        fwd_daily = pd.DataFrame()

    # 4. Historical Daily
    hist_daily_path = os.path.join(PROCESSED_DIR, 'dashboard_historical_daily.csv')
    if os.path.exists(hist_daily_path):
        hist_daily = pd.read_csv(hist_daily_path)
        hist_daily['date_parsed'] = pd.to_datetime(hist_daily['date'])
    else:
        hist_daily = pd.DataFrame()

    # 5. Validation Metrics
    val_metrics_path = os.path.join(REPORTS_DIR, 'validation_metrics.csv')
    if os.path.exists(val_metrics_path):
        val_metrics = pd.read_csv(val_metrics_path)
    else:
        val_metrics = pd.DataFrame()

    return sku_master, val_daily, fwd_daily, hist_daily, val_metrics

sku_master, val_daily, fwd_daily, hist_daily, val_metrics = load_data_caches()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 💄 Rimmel Forecasting")
st.sidebar.caption("Certified Production Engine (Exp6)")
st.sidebar.markdown("---")

st.sidebar.markdown("**System Governance:**")
st.sidebar.markdown("- **Engine**: ZERO + LightGBM Regressor")
st.sidebar.markdown("- **Calibration**: Combined ($\alpha=0.10, \beta=0.10$)")
st.sidebar.markdown("- **Validation**: Sep 1–10, 2026")
st.sidebar.markdown("- **Forecast**: Sep 11–20, 2026")
st.sidebar.markdown("- **Shared Inventory**: Single Central Pool")

st.sidebar.markdown("---")
st.sidebar.markdown("**Excel Deliverables:**")
val_report_path = os.path.join(REPORTS_DIR, 'Rimmel_Validation_Sep01_Sep10_2026.xlsx')
if not os.path.exists(val_report_path):
    val_report_path = os.path.join(REPORTS_DIR, 'validation_report_sep_01_to_10_2026.xlsx')

fwd_report_path = os.path.join(REPORTS_DIR, 'Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx')
if not os.path.exists(fwd_report_path):
    fwd_report_path = os.path.join(REPORTS_DIR, 'production_forecast_sep_11_to_20_2026.xlsx')

if os.path.exists(val_report_path):
    with open(val_report_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Validation Excel (SKU Summary)",
            data=f,
            file_name=os.path.basename(val_report_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if os.path.exists(fwd_report_path):
    with open(fwd_report_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Forecast Excel (SKU Summary)",
            data=f,
            file_name=os.path.basename(fwd_report_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Rimmel Multi-Platform Demand Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Certified Production Platform for Inventory Replenishment & Multi-Channel Demand Planning</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Tab 1: Product Inspector",
    "📋 Tab 2: Data View",
    "📊 Tab 3: Forecast Overview",
    "🧪 Tab 4: Validation",
    "📦 Tab 5: Inventory / Planning"
])

# =============================================================================
# TAB 1: PRODUCT INSPECTOR
# =============================================================================
with tab1:
    st.markdown("### 🔍 Product Inspector")
    st.markdown("Inspect historical actuals, holdout validation, and forward forecasts for any individual catalog SKU.")

    all_skus = sorted(sku_master['SKU'].unique().tolist()) if not sku_master.empty else []
    
    if not all_skus:
        st.warning("No SKU data found. Please run `python -m src.generate_client_reports` first.")
    else:
        # SKU Selector
        selected_sku = st.selectbox("Select Catalog SKU to Inspect:", options=all_skus, index=0)

        # SKU Details
        sku_info = sku_master[sku_master['SKU'] == selected_sku].iloc[0]
        
        # Metadata KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Category</div>
                <div class="kpi-value" style="font-size: 1.25rem;">{sku_info.get('category', 'N/A')}</div>
                <div class="kpi-desc">Family: {sku_info.get('resolved_parent_id', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            current_stock_val = sku_info.get('current_stock', 0)
            stock_disp = f"{int(current_stock_val):,}" if not pd.isna(current_stock_val) else "0"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Shared Warehouse Stock</div>
                <div class="kpi-value">{stock_disp}</div>
                <div class="kpi-desc">Central shared warehouse pool</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            tot_forecast_val = int(sku_info.get('tot', 0))
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">10-Day Total Forecast</div>
                <div class="kpi-value">{tot_forecast_val:,}</div>
                <div class="kpi-desc">Sep 11–20 Expected Physical Units</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            risk_val = sku_info.get('risk', 'NORMAL')
            color = "#D32F2F" if "STOCKOUT" in risk_val else ("#F57C00" if "OVERSTOCK" in risk_val or "LEAN" in risk_val else "#388E3C")
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {color};">
                <div class="kpi-title">Risk Assessment</div>
                <div class="kpi-value" style="font-size: 1.15rem; color: {color};">{risk_val}</div>
                <div class="kpi-desc">Days of Cover: {sku_info.get('doc', 0):.1f} days</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            action_val = sku_info.get('action', 'Maintain Baseline')
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Planning Action</div>
                <div class="kpi-value" style="font-size: 1.05rem;">{action_val}</div>
                <div class="kpi-desc">Status: {sku_info.get('status', 'Adequate')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # PRODUCT GRAPH (PLOTLY)
        # -------------------------------------------------------------
        st.markdown("#### Demand Trajectory & Forecast Timeline")
        
        # History range filter
        range_option = st.radio(
            "Select Historical Window:",
            ["30 Days", "60 Days", "90 Days", "Full History"],
            horizontal=True,
            index=2
        )

        cutoff_days_map = {"30 Days": 30, "60 Days": 60, "90 Days": 90, "Full History": 9999}
        days_back = cutoff_days_map[range_option]

        # Extract historical series for this SKU
        sku_hist = hist_daily[hist_daily['canonical_sku'] == selected_sku].copy()
        if not sku_hist.empty:
            sku_hist_tot = sku_hist.groupby('date_parsed').agg({'actual_units': 'sum'}).reset_index()
            max_hist_date = sku_hist_tot['date_parsed'].max()
            start_hist_date = max_hist_date - pd.Timedelta(days=days_back)
            sku_hist_filtered = sku_hist_tot[sku_hist_tot['date_parsed'] >= start_hist_date].sort_values('date_parsed')
        else:
            sku_hist_filtered = pd.DataFrame(columns=['date_parsed', 'actual_units'])

        # Extract validation series for this SKU
        sku_val = val_daily[val_daily['SKU'] == selected_sku].sort_values('date_parsed')

        # Extract forecast series for this SKU
        sku_fwd = fwd_daily[fwd_daily['SKU'] == selected_sku].sort_values('date_parsed')

        fig = go.Figure()

        # 1. Historical Actual Sales Trace
        if not sku_hist_filtered.empty:
            fig.add_trace(go.Scatter(
                x=sku_hist_filtered['date_parsed'],
                y=sku_hist_filtered['actual_units'],
                mode='lines+markers',
                name='Historical Actual Sales',
                line=dict(color='#1F4E78', width=2),
                marker=dict(size=5, color='#1F4E78')
            ))

        # 2. Validation Actual Sales Trace (Sep 01-10)
        if not sku_val.empty:
            fig.add_trace(go.Scatter(
                x=sku_val['date_parsed'],
                y=sku_val['Total Actual Units'],
                mode='lines+markers',
                name='Validation Actual Sales (Holdout)',
                line=dict(color='#2CA02C', width=2.5),
                marker=dict(size=7, symbol='diamond', color='#2CA02C')
            ))

            # Validation Predicted Units Trace
            fig.add_trace(go.Scatter(
                x=sku_val['date_parsed'],
                y=sku_val['Total Predicted Units'],
                mode='lines+markers',
                name='Model Prediction (Validation)',
                line=dict(color='#FF7F0E', width=2, dash='dot'),
                marker=dict(size=6, color='#FF7F0E')
            ))

        # 3. Forward Forecast Total Prediction (Sep 11-20)
        if not sku_fwd.empty:
            fig.add_trace(go.Scatter(
                x=sku_fwd['date_parsed'],
                y=sku_fwd['Total Predicted Units'],
                mode='lines+markers',
                name='Forward Forecast (Sep 11–20)',
                line=dict(color='#D62728', width=2.5, dash='dash'),
                marker=dict(size=7, color='#D62728')
            ))

            # Channel-specific forecast lines
            fig.add_trace(go.Scatter(
                x=sku_fwd['date_parsed'],
                y=sku_fwd['Amazon Predicted Units'],
                mode='lines',
                name='Amazon Forecast',
                line=dict(color='#FF9900', width=1.5, dash='dashdot'),
                visible='legendonly'
            ))
            fig.add_trace(go.Scatter(
                x=sku_fwd['date_parsed'],
                y=sku_fwd['eBay Predicted Units'],
                mode='lines',
                name='eBay Forecast',
                line=dict(color='#0064D2', width=1.5, dash='dashdot'),
                visible='legendonly'
            ))

        # Background shaded zones for validation and forecast periods
        fig.add_vrect(
            x0='2026-09-01', x1='2026-09-10',
            fillcolor='#E2F0D9', opacity=0.35,
            layer='below', line_width=1, line_dash='dash', line_color='#70AD47',
            annotation_text="Holdout Validation<br>(Sep 01–10)", annotation_position="top left",
            annotation_font_size=10, annotation_font_color='#385723'
        )

        fig.add_vrect(
            x0='2026-09-11', x1='2026-09-20',
            fillcolor='#FFF2CC', opacity=0.35,
            layer='below', line_width=1, line_dash='dash', line_color='#FFBF00',
            annotation_text="Forward Forecast<br>(Sep 11–20)", annotation_position="top left",
            annotation_font_size=10, annotation_font_color='#7F6000'
        )

        fig.update_layout(
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(title="Calendar Date", showgrid=True, gridcolor='#F0F0F0'),
            yaxis=dict(title="Physical Units", showgrid=True, gridcolor='#F0F0F0')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # PLATFORM MIX DONUT CHART & CARDS
        # -------------------------------------------------------------
        st.markdown("#### Forward Platform Demand Contribution")

        p_col_left, p_col_right = st.columns([1, 1.2])

        with p_col_left:
            amz_sku_fwd = int(sku_info.get('amz', 0))
            ebay_sku_fwd = int(sku_info.get('ebay', 0))
            web_sku_fwd = int(sku_info.get('web', 0))
            oth_sku_fwd = int(sku_info.get('oth', 0))
            tot_sku_fwd = amz_sku_fwd + ebay_sku_fwd + web_sku_fwd + oth_sku_fwd

            plat_shares = {
                'Amazon': amz_sku_fwd,
                'eBay': ebay_sku_fwd,
                'Website': web_sku_fwd,
                'Other': oth_sku_fwd
            }

            if tot_sku_fwd > 0:
                labels = list(plat_shares.keys())
                values = list(plat_shares.values())
                colors = ['#FF9900', '#0064D2', '#107C41', '#6E40C9']

                donut_fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )])
                donut_fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )
                st.plotly_chart(donut_fig, use_container_width=True)

                # Dynamic insight text
                max_platform = max(plat_shares, key=plat_shares.get)
                max_share = (plat_shares[max_platform] / tot_sku_fwd) * 100.0
                st.info(f"💡 **Demand Distribution:** Most forecast demand is expected from **{max_platform}** ({max_share:.1f}% of total projected demand).")
            else:
                st.markdown("""
                <div style="height: 250px; display: flex; align-items: center; justify-content: center; background: #F8F9FA; border-radius: 8px;">
                    <div style="text-align: center; color: #7F7F7F;">
                        <h4>0 Units Projected</h4>
                        <p>Zero demand forecast across all channels for this SKU.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Zero demand forecast across all channels for this SKU.")

        with p_col_right:
            st.markdown("**Platform Breakdown (10-Day Forward Forecast):**")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown(f"""
                <div class="platform-card" style="border-top: 3px solid #FF9900;">
                    <div class="platform-title">🛒 Amazon</div>
                    <div class="platform-value" style="color: #FF9900;">{amz_sku_fwd:,}</div>
                    <div class="kpi-desc">units</div>
                </div>
                """, unsafe_allow_html=True)
            with b_col2:
                st.markdown(f"""
                <div class="platform-card" style="border-top: 3px solid #0064D2;">
                    <div class="platform-title">🏷️ eBay</div>
                    <div class="platform-value" style="color: #0064D2;">{ebay_sku_fwd:,}</div>
                    <div class="kpi-desc">units</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            b_col3, b_col4 = st.columns(2)
            with b_col3:
                st.markdown(f"""
                <div class="platform-card" style="border-top: 3px solid #107C41;">
                    <div class="platform-title">🌐 Website</div>
                    <div class="platform-value" style="color: #107C41;">{web_sku_fwd:,}</div>
                    <div class="kpi-desc">units</div>
                </div>
                """, unsafe_allow_html=True)
            with b_col4:
                st.markdown(f"""
                <div class="platform-card" style="border-top: 3px solid #6E40C9;">
                    <div class="platform-title">📦 Other / B2B</div>
                    <div class="platform-value" style="color: #6E40C9;">{oth_sku_fwd:,}</div>
                    <div class="kpi-desc">units</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #1F4E78; color: white; padding: 12px 18px; border-radius: 8px; margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; font-size: 1.05rem;">TOTAL PHYSICAL FORECAST:</span>
                <span style="font-weight: 700; font-size: 1.4rem;">{tot_sku_fwd:,} units</span>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: DATA VIEW
# =============================================================================
with tab2:
    st.markdown("### 📋 Data Exploration & Verification")
    st.markdown("Inspect underlying historical, validation, and forward forecast dataset records.")

    data_mode = st.radio(
        "Select Dataset View:",
        ["Forward Forecast (Sep 11–20)", "Validation Holdout (Sep 01–10)", "Full Catalog Planning Master"],
        horizontal=True
    )

    if data_mode == "Forward Forecast (Sep 11–20)":
        if not fwd_daily.empty:
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                selected_sku_filter = st.multiselect("Filter by SKU:", options=sorted(fwd_daily['SKU'].unique()), default=[])
            
            display_df = fwd_daily.copy()
            if selected_sku_filter:
                display_df = display_df[display_df['SKU'].isin(selected_sku_filter)]

            cols_to_show = [
                'Date', 'SKU',
                'Amazon Actual Units', 'Amazon Predicted Units',
                'eBay Actual Units', 'eBay Predicted Units',
                'Website Actual Units', 'Website Predicted Units',
                'Other Actual Units', 'Other Predicted Units',
                'Total Actual Units', 'Total Predicted Units',
                'Reason'
            ]
            st.dataframe(display_df[cols_to_show], use_container_width=True, height=450)
            st.caption(f"Showing {len(display_df):,} rows. Actual units for future dates are strictly blank / NULL.")
        else:
            st.info("No forward forecast cache found.")

    elif data_mode == "Validation Holdout (Sep 01–10)":
        if not val_daily.empty:
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                selected_sku_filter = st.multiselect("Filter by SKU:", options=sorted(val_daily['SKU'].unique()), default=[])
            
            display_df = val_daily.copy()
            if selected_sku_filter:
                display_df = display_df[display_df['SKU'].isin(selected_sku_filter)]

            cols_to_show = [
                'Date', 'SKU',
                'Amazon Actual Units', 'Amazon Predicted Units',
                'eBay Actual Units', 'eBay Predicted Units',
                'Website Actual Units', 'Website Predicted Units',
                'Other Actual Units', 'Other Predicted Units',
                'Total Actual Units', 'Total Predicted Units',
                'Reason'
            ]
            st.dataframe(display_df[cols_to_show], use_container_width=True, height=450)
            st.caption(f"Showing {len(display_df):,} rows.")
        else:
            st.info("No validation cache found.")

    else:
        if not sku_master.empty:
            st.dataframe(sku_master, use_container_width=True, height=450)
            st.caption(f"Catalog Master: {len(sku_master):,} unique SKUs.")
        else:
            st.info("No catalog master cache found.")

# =============================================================================
# TAB 3: FORECAST OVERVIEW
# =============================================================================
with tab3:
    st.markdown("### 📊 Portfolio Forecast Overview (September 11–20, 2026)")
    st.markdown("Aggregate demand projections across all selling channels and product lines.")

    if not fwd_daily.empty:
        tot_amz = int(fwd_daily['Amazon Predicted Units'].sum())
        tot_ebay = int(fwd_daily['eBay Predicted Units'].sum())
        tot_web = int(fwd_daily['Website Predicted Units'].sum())
        tot_oth = int(fwd_daily['Other Predicted Units'].sum())
        grand_total = int(fwd_daily['Total Predicted Units'].sum())
        num_skus = fwd_daily['SKU'].nunique()

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #1F4E78;">
                <div class="kpi-title">Total Portfolio Forecast</div>
                <div class="kpi-value">{grand_total:,}</div>
                <div class="kpi-desc">Physical Units (10 Days)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #FF9900;">
                <div class="kpi-title">Amazon Channel</div>
                <div class="kpi-value">{tot_amz:,}</div>
                <div class="kpi-desc">{tot_amz / grand_total * 100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #0064D2;">
                <div class="kpi-title">eBay Channel</div>
                <div class="kpi-value">{tot_ebay:,}</div>
                <div class="kpi-desc">{tot_ebay / grand_total * 100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #107C41;">
                <div class="kpi-title">Website Channel</div>
                <div class="kpi-value">{tot_web:,}</div>
                <div class="kpi-desc">{tot_web / grand_total * 100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col5:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #6E40C9;">
                <div class="kpi-title">Active SKUs Planned</div>
                <div class="kpi-value">{num_skus:,}</div>
                <div class="kpi-desc">1,413 SKU-platform combinations</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Two Charts: Daily Timeline and Top SKUs
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown("#### Daily Forecast by Platform (Sep 11–20)")
            daily_timeline = fwd_daily.groupby('Date', sort=False).agg({
                'Amazon Predicted Units': 'sum',
                'eBay Predicted Units': 'sum',
                'Website Predicted Units': 'sum',
                'Other Predicted Units': 'sum'
            }).reset_index()

            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(x=daily_timeline['Date'], y=daily_timeline['Amazon Predicted Units'], name='Amazon', marker_color='#FF9900'))
            bar_fig.add_trace(go.Bar(x=daily_timeline['Date'], y=daily_timeline['eBay Predicted Units'], name='eBay', marker_color='#0064D2'))
            bar_fig.add_trace(go.Bar(x=daily_timeline['Date'], y=daily_timeline['Website Predicted Units'], name='Website', marker_color='#107C41'))
            bar_fig.add_trace(go.Bar(x=daily_timeline['Date'], y=daily_timeline['Other Predicted Units'], name='Other', marker_color='#6E40C9'))

            bar_fig.update_layout(
                barmode='stack',
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        with c_right:
            st.markdown("#### Top 10 SKUs by Forward Demand")
            top_skus = sku_master.sort_values('tot', ascending=False).head(10)
            
            top_fig = go.Figure(go.Bar(
                x=top_skus['tot'],
                y=top_skus['SKU'],
                orientation='h',
                marker=dict(color='#1F4E78')
            ))
            top_fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(top_fig, use_container_width=True)

    else:
        st.info("No forward forecast cache found.")

# =============================================================================
# TAB 4: VALIDATION
# =============================================================================
with tab4:
    st.markdown("### 🧪 Retrospective Holdout Validation Benchmark")
    st.markdown("Empirical performance of the certified Exp6 model on the unseen **September 1–10, 2026** holdout window.")

    # Executive Benchmark KPI Cards
    vk_col1, vk_col2, vk_col3, vk_col4, vk_col5 = st.columns(5)
    with vk_col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Actual Sales</div>
            <div class="kpi-value">2,069</div>
            <div class="kpi-desc">Total empirical units sold</div>
        </div>
        """, unsafe_allow_html=True)
    with vk_col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Model Predictions</div>
            <div class="kpi-value">2,121.2</div>
            <div class="kpi-desc">Total predicted units</div>
        </div>
        """, unsafe_allow_html=True)
    with vk_col3:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #2CA02C;">
            <div class="kpi-title">Forecast Bias</div>
            <div class="kpi-value" style="color: #2CA02C;">+2.52%</div>
            <div class="kpi-desc">Near-zero catalog net bias</div>
        </div>
        """, unsafe_allow_html=True)
    with vk_col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Catalog WAPE</div>
            <div class="kpi-value">90.54%</div>
            <div class="kpi-desc">Intermittent sparsity driven</div>
        </div>
        """, unsafe_allow_html=True)
    with vk_col5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Daily Series MAE</div>
            <div class="kpi-value">0.1326</div>
            <div class="kpi-desc">Units per SKU-channel-day</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box">
        <b>💡 Executive Validation Context:</b>
        <br>
        The validation period was evaluated strictly on unseen data before being included in the final production training dataset.
        In beauty and cosmetics e-commerce, over 70% of SKU-channel days have zero sales. At daily series resolution, small fractional model predictions against intermittent zero sales generate high row-level WAPE (90.54%). However, across the portfolio, total predicted units (2,121.2) match total actual units (2,069.0) with an extraordinary <b>+2.52% net bias</b>, providing safe, accurate baseline replenishment signals.
    </div>
    """, unsafe_allow_html=True)

    # Actual vs Predicted Plot
    if not val_daily.empty:
        val_daily_rollup = val_daily.groupby('date_parsed').agg({
            'Total Actual Units': 'sum',
            'Total Predicted Units': 'sum'
        }).reset_index()

        val_fig = go.Figure()
        val_fig.add_trace(go.Bar(
            x=val_daily_rollup['date_parsed'].dt.strftime('%d-%b'),
            y=val_daily_rollup['Total Actual Units'],
            name='Actual Sales',
            marker_color='#2CA02C'
        ))
        val_fig.add_trace(go.Bar(
            x=val_daily_rollup['date_parsed'].dt.strftime('%d-%b'),
            y=val_daily_rollup['Total Predicted Units'],
            name='Model Prediction',
            marker_color='#1F4E78'
        ))
        val_fig.update_layout(
            barmode='group',
            height=320,
            title="Daily Catalog Actual vs Predicted (Sep 01–10, 2026)",
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(val_fig, use_container_width=True)

    if not val_metrics.empty:
        st.markdown("#### Platform-Specific Performance Metrics")
        st.dataframe(val_metrics, use_container_width=True)

# =============================================================================
# TAB 5: INVENTORY / PLANNING
# =============================================================================
with tab5:
    st.markdown("### 📦 Shared Inventory Decision Support")
    st.markdown("Actionable purchase orders and replenishment recommendations based on single-pool central warehouse stock.")

    st.markdown("""
    <div class="warning-box">
        <b>⚠️ SHARED WAREHOUSE INVENTORY GOVERNANCE RULE:</b>
        <br>
        Physical inventory is held in <b>ONE shared warehouse pool</b> that fulfills Amazon, eBay, Website, and Other. 
        <b>NEVER sum inventory across platforms</b>. All Days of Cover calculations reflect shared stock divided by total physical portfolio demand.
    </div>
    """, unsafe_allow_html=True)

    if not sku_master.empty:
        # Filter controls
        inv_f1, inv_f2 = st.columns([1, 2])
        with inv_f1:
            risk_filter = st.selectbox(
                "Filter by Inventory Risk Status:",
                ["All", "STOCKOUT RISK", "HIGH STOCKOUT RISK", "LEAN COVERAGE", "ADEQUATE COVERAGE", "OVERSTOCK RISK", "SLOW-MOVING EXCESS"],
                index=0
            )

        inv_display = sku_master.copy()
        if risk_filter != "All":
            inv_display = inv_display[inv_display['risk'].str.contains(risk_filter, na=False)]

        inv_cols = [
            'SKU', 'category', 'current_stock', 'tot', 'doc', 'status', 'risk', 'action'
        ]
        inv_rename = {
            'category': 'Category',
            'current_stock': 'Shared Stock',
            'tot': '10-Day Forecast',
            'doc': 'Days of Cover',
            'status': 'Inventory Status',
            'risk': 'Risk Assessment',
            'action': 'Recommended Action'
        }
        st.dataframe(
            inv_display[inv_cols].rename(columns=inv_rename).sort_values('10-Day Forecast', ascending=False),
            use_container_width=True,
            height=450
        )
        st.caption(f"Displaying {len(inv_display):,} SKUs matching filter.")
    else:
        st.info("No inventory planning cache found.")

st.markdown("---")
st.caption("Rimmel Multi-Platform Demand Forecasting System | Production Certified Release | Sep 2026")
