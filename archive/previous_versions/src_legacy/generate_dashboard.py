"""
Generate Comprehensive Production HTML Dashboard
Includes 1.5-Year Historical Trends, 3-Tier Forecast Preview, Category Breakdown,
Price vs Volume Scatter, Day-of-Week Radar, and Full Scrollable Predictions Table.
"""
import sqlite3, os, json
import pandas as pd
import numpy as np

BASE_DIR    = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH     = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
OUT_HTML    = os.path.join(REPORTS_DIR, 'Rimmel_Dashboard.html')

os.makedirs(REPORTS_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

df_full = pd.read_sql("SELECT * FROM full_history", conn)

# Try reading from 3-tier master forecast first, then fallbacks
try:
    df_fore = pd.read_sql("SELECT * FROM forecast_aug_2026_3tier_master", conn)
except Exception:
    try:
        df_fore = pd.read_sql("SELECT * FROM forecast_aug_2026_1year_pattern", conn)
    except Exception:
        df_fore = pd.read_sql("SELECT * FROM forecast_aug_2026", conn)

conn.close()

df_full['date'] = pd.to_datetime(df_full['date'])

# Dynamic SKU Summary calculation
df_smry = df_full.groupby('sku').agg(
    total_units=('total_sales', 'sum'),
    avg_price=('selling_price', 'median'),
    avg_daily_sales=('total_sales', 'mean'),
    category=('category', 'first')
).reset_index()

# ── Aggregations ──────────────────────────────────────────────────────────────

# 1. Monthly total sales (Jan 2025 - Jul 2026)
df_full['month_label'] = df_full['date'].dt.to_period('M').astype(str)
monthly = df_full.groupby('month_label')['total_sales'].sum().reset_index()
monthly = monthly.sort_values('month_label')

# 2. Top 20 SKUs by total sales
top20 = df_smry.sort_values('total_units', ascending=False).head(20)

# 3. Category breakdown
cat = df_full.groupby('category')['total_sales'].sum().reset_index()
cat = cat.sort_values('total_sales', ascending=False).head(10)

# 4. Weekly sales trend
df_full['week'] = df_full['date'].dt.to_period('W').astype(str)
weekly = df_full.groupby('week')['total_sales'].sum().reset_index().tail(78)

# 5. Top 10 forecast
top10_fore = df_fore.sort_values('predicted unit', ascending=False).head(10)

# 6. Day-of-week pattern
df_full['dow_name'] = df_full['date'].dt.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = df_full.groupby('dow_name')['total_sales'].mean().reindex(dow_order).reset_index()

# 7. Price vs Sales scatter (top 50 SKUs)
scatter_data = df_smry.sort_values('total_units', ascending=False).head(50)[
    ['sku','avg_price','avg_daily_sales','total_units','category']
].dropna()

# 8. KPIs
total_units  = int(df_full['total_sales'].sum())
total_skus   = int(df_full['sku'].nunique())
best_sku     = df_smry.sort_values('total_units', ascending=False).iloc[0]
forecast_tot = int(df_fore['predicted unit'].sum())
avg_daily    = round(df_full.groupby('date')['total_sales'].sum().mean(), 1)

# ── JSON for Charts ───────────────────────────────────────────────────────────
monthly_labels = monthly['month_label'].tolist()
monthly_vals   = monthly['total_sales'].tolist()

weekly_labels = weekly['week'].tolist()
weekly_vals   = weekly['total_sales'].tolist()

top20_labels = top20['sku'].tolist()
top20_vals   = top20['total_units'].tolist()

cat_labels = cat['category'].tolist()
cat_vals   = cat['total_sales'].tolist()

dow_labels = dow['dow_name'].tolist()
dow_vals   = [round(v,1) for v in dow['total_sales'].tolist()]

fore_labels = top10_fore['order sku'].tolist()
fore_pred   = top10_fore['predicted unit'].tolist()
fore_lower  = top10_fore['lower bound'].tolist()
fore_upper  = top10_fore['upper bound'].tolist()

scatter_pts = [
    {'x': round(float(r['avg_price']),2),
     'y': round(float(r['avg_daily_sales']),2),
     'r': max(4, min(30, float(r['total_units'])/200)),
     'sku': r['sku'],
     'total': int(r['total_units'])}
    for _, r in scatter_data.iterrows()
]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rimmel Sales Intelligence Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;background:#0a0f1e;color:#e2e8f0;min-height:100vh}}
  
  /* Header */
  .header{{background:linear-gradient(135deg,#1a237e 0%,#0d47a1 50%,#006064 100%);
           padding:28px 36px;display:flex;align-items:center;justify-content:space-between;
           box-shadow:0 4px 30px rgba(0,0,0,0.4)}}
  .header-left h1{{font-size:26px;font-weight:800;color:#fff;letter-spacing:-0.5px}}
  .header-left p{{font-size:13px;color:#90caf9;margin-top:4px}}
  .badge{{background:rgba(255,255,255,0.15);backdrop-filter:blur(10px);
          border:1px solid rgba(255,255,255,0.2);border-radius:8px;
          padding:8px 16px;font-size:12px;font-weight:600;color:#fff}}
  
  /* KPI Row */
  .kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;
            padding:24px 36px 0}}
  .kpi{{background:linear-gradient(135deg,#111827,#1e2a3a);
        border:1px solid rgba(255,255,255,0.07);border-radius:14px;
        padding:20px;position:relative;overflow:hidden}}
  .kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
  .kpi:nth-child(1)::before{{background:linear-gradient(90deg,#3b82f6,#06b6d4)}}
  .kpi:nth-child(2)::before{{background:linear-gradient(90deg,#10b981,#34d399)}}
  .kpi:nth-child(3)::before{{background:linear-gradient(90deg,#f59e0b,#fbbf24)}}
  .kpi:nth-child(4)::before{{background:linear-gradient(90deg,#8b5cf6,#a78bfa)}}
  .kpi:nth-child(5)::before{{background:linear-gradient(90deg,#ef4444,#f87171)}}
  .kpi-label{{font-size:10px;font-weight:700;text-transform:uppercase;
              letter-spacing:1px;color:#64748b;margin-bottom:10px}}
  .kpi-value{{font-size:28px;font-weight:800;color:#f1f5f9;line-height:1}}
  .kpi-sub{{font-size:11px;color:#94a3b8;margin-top:6px}}
  
  /* Grid layout */
  .grid{{display:grid;gap:16px;padding:16px 36px}}
  .grid-2{{grid-template-columns:1fr 1fr}}
  .grid-3{{grid-template-columns:2fr 1fr}}
  .grid-full{{grid-template-columns:1fr}}
  
  /* Chart cards */
  .card{{background:linear-gradient(135deg,#111827,#1e2a3a);
         border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:24px}}
  .card-title{{font-size:13px;font-weight:700;text-transform:uppercase;
               letter-spacing:0.8px;color:#94a3b8;margin-bottom:4px}}
  .card-subtitle{{font-size:11px;color:#475569;margin-bottom:18px}}
  .chart-wrap{{position:relative}}
  
  /* Table */
  .table-scroll{{max-height:360px;overflow-y:auto}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead th{{background:#0f172a;color:#94a3b8;font-weight:700;text-transform:uppercase;
            font-size:10px;letter-spacing:0.8px;padding:10px 12px;
            position:sticky;top:0;border-bottom:1px solid rgba(255,255,255,0.06)}}
  tbody tr{{border-bottom:1px solid rgba(255,255,255,0.04);transition:background 0.15s}}
  tbody tr:hover{{background:rgba(59,130,246,0.08)}}
  td{{padding:9px 12px;color:#cbd5e1}}
  td:nth-child(3),td:nth-child(4),td:nth-child(5){{text-align:right;font-weight:600}}
  .rank{{color:#64748b;width:30px}}
  .sku-code{{font-weight:700;color:#60a5fa;font-size:11px}}
  .bar-wrap{{display:flex;align-items:center;gap:8px}}
  .mini-bar{{height:6px;background:linear-gradient(90deg,#3b82f6,#06b6d4);
             border-radius:3px;min-width:4px;transition:width 0.3s}}
  
  /* Footer */
  .footer{{text-align:center;padding:24px;color:#334155;font-size:11px;
           border-top:1px solid rgba(255,255,255,0.05);margin-top:8px}}
  
  /* Scrollbar */
  ::-webkit-scrollbar{{width:6px;height:6px}}
  ::-webkit-scrollbar-track{{background:#0a0f1e}}
  ::-webkit-scrollbar-thumb{{background:#334155;border-radius:3px}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>&#128202; Rimmel Sales Intelligence Dashboard</h1>
    <p>1.5 Years Historical Analysis + August 2026 Forecast &nbsp;|&nbsp; 3-Tier Production Engine Active</p>
  </div>
  <div class="badge">&#128197; Aug 1 &ndash; Aug 11, 2026 Forecast Active</div>
</div>

<!-- KPI ROW -->
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Total Units Sold</div>
    <div class="kpi-value">{total_units:,}</div>
    <div class="kpi-sub">Jan 2025 &ndash; Jul 2026 (1.5 years)</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Active SKUs</div>
    <div class="kpi-value">{total_skus}</div>
    <div class="kpi-sub">Unique products tracked</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Daily Sales</div>
    <div class="kpi-value">{avg_daily:,.0f}</div>
    <div class="kpi-sub">Units per day (all SKUs)</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Aug Forecast Total</div>
    <div class="kpi-value">{forecast_tot:,}</div>
    <div class="kpi-sub">Aug 1&ndash;11 across active SKUs</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best Seller</div>
    <div class="kpi-value" style="font-size:16px">{best_sku['sku']}</div>
    <div class="kpi-sub">{int(best_sku['total_units']):,} units in 1.5 years</div>
  </div>
</div>

<!-- ROW 1: Monthly Trend -->
<div class="grid grid-full">
  <div class="card">
    <div class="card-title">Monthly Sales Trend</div>
    <div class="card-subtitle">Total units sold per month across all SKUs (Jan 2025 &ndash; Jul 2026)</div>
    <div class="chart-wrap"><canvas id="monthlyChart" height="90"></canvas></div>
  </div>
</div>

<!-- ROW 2: Top 20 SKUs + Day-of-Week -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Top 20 SKUs by Total Sales</div>
    <div class="card-subtitle">Total units sold Aug 2025 &ndash; Jul 2026 (training window)</div>
    <div class="chart-wrap"><canvas id="top20Chart" height="220"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Sales by Day of Week</div>
    <div class="card-subtitle">Average daily units per weekday across full history</div>
    <div class="chart-wrap"><canvas id="dowChart" height="220"></canvas></div>
  </div>
</div>

<!-- ROW 3: Category Donut + Price vs Volume Scatter -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Sales by Product Category</div>
    <div class="card-subtitle">Top 10 categories — share of total units</div>
    <div class="chart-wrap"><canvas id="catChart" height="200"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Price vs. Daily Sales Volume</div>
    <div class="card-subtitle">Top 50 SKUs — bubble size = total volume</div>
    <div class="chart-wrap"><canvas id="scatterChart" height="200"></canvas></div>
  </div>
</div>

<!-- ROW 4: August Forecast Bar + Forecast Table -->
<div class="grid grid-3">
  <div class="card">
    <div class="card-title">August 1&ndash;11 Forecast — Top 10 SKUs</div>
    <div class="card-subtitle">Predicted units with confidence interval (lower / upper bound)</div>
    <div class="chart-wrap"><canvas id="forecastChart" height="230"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Weekly Sales Trend</div>
    <div class="card-subtitle">Last 78 weeks rolling total (all SKUs)</div>
    <div class="chart-wrap"><canvas id="weeklyChart" height="230"></canvas></div>
  </div>
</div>

<!-- ROW 5: Full Forecast Table -->
<div class="grid grid-full" style="padding-bottom:8px">
  <div class="card">
    <div class="card-title">Full Forecast Table — All SKUs (Aug 1&ndash;11, 2026)</div>
    <div class="card-subtitle">Sorted by predicted units. Scroll to explore all product forecasts.</div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th class="rank">#</th>
            <th>SKU</th>
            <th>Predicted Units</th>
            <th>Lower Bound</th>
            <th>Upper Bound</th>
            <th>Tier Engine</th>
            <th>Volume Bar</th>
          </tr>
        </thead>
        <tbody id="forecastTableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="footer">
  Rimmel Sales Intelligence Platform &nbsp;&bull;&nbsp; 3-Tier Hybrid Production Engine &nbsp;&bull;&nbsp; 
  Forecast Window: Aug 1&ndash;11 2026 &nbsp;&bull;&nbsp; Data: Multi-Channel Aggregated Clean Database
</div>

<script>
Chart.register(ChartDataLabels);

const PALETTE = [
  '#3b82f6','#06b6d4','#10b981','#f59e0b','#8b5cf6',
  '#ef4444','#f97316','#84cc16','#ec4899','#14b8a6',
  '#a78bfa','#fb923c','#34d399','#fbbf24','#60a5fa',
  '#67e8f9','#6ee7b7','#fcd34d','#c084fc','#f9a8d4'
];

const gridColor = 'rgba(255,255,255,0.06)';
const textColor = '#94a3b8';
const baseOptions = {{
  responsive: true,
  plugins: {{
    legend: {{ labels: {{ color: textColor, font: {{ size: 11 }} }} }},
    datalabels: {{ display: false }}
  }},
  scales: {{
    x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, font: {{ size: 10 }} }} }},
    y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, font: {{ size: 10 }} }} }}
  }}
}};

// 1. Monthly Trend
new Chart(document.getElementById('monthlyChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(monthly_labels)},
    datasets: [{{
      label: 'Total Units Sold',
      data: {json.dumps(monthly_vals)},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.15)',
      borderWidth: 2.5,
      pointBackgroundColor: '#3b82f6',
      pointRadius: 4,
      fill: true,
      tension: 0.4
    }}]
  }},
  options: {{
    ...baseOptions,
    plugins: {{ ...baseOptions.plugins,
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.y.toLocaleString() + ' units' }} }} }},
    scales: {{ ...baseOptions.scales,
      y: {{ ...baseOptions.scales.y,
        ticks: {{ color: textColor, callback: v => v.toLocaleString() }} }} }}
  }}
}});

// 2. Top 20 SKUs Horizontal Bar
new Chart(document.getElementById('top20Chart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(top20_labels)},
    datasets: [{{
      label: 'Total Units (1.5 years)',
      data: {json.dumps(top20_vals)},
      backgroundColor: PALETTE,
      borderRadius: 5,
      borderSkipped: false
    }}]
  }},
  options: {{
    ...baseOptions,
    indexAxis: 'y',
    plugins: {{ ...baseOptions.plugins,
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.x.toLocaleString() + ' units' }} }} }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toLocaleString() }} }},
      y: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#60a5fa', font: {{ size: 10, weight: '600' }} }} }}
    }}
  }}
}});

// 3. Day-of-Week Radar
new Chart(document.getElementById('dowChart'), {{
  type: 'radar',
  data: {{
    labels: {json.dumps(dow_labels)},
    datasets: [{{
      label: 'Avg Daily Units',
      data: {json.dumps(dow_vals)},
      borderColor: '#10b981',
      backgroundColor: 'rgba(16,185,129,0.2)',
      pointBackgroundColor: '#10b981',
      pointRadius: 5,
      borderWidth: 2
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: textColor }} }},
      datalabels: {{ display: false }}
    }},
    scales: {{
      r: {{
        grid: {{ color: gridColor }},
        pointLabels: {{ color: textColor, font: {{ size: 11 }} }},
        ticks: {{ color: textColor, backdropColor: 'transparent', font: {{ size: 9 }} }}
      }}
    }}
  }}
}});

// 4. Category Donut
new Chart(document.getElementById('catChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(cat_labels)},
    datasets: [{{
      data: {json.dumps(cat_vals)},
      backgroundColor: PALETTE,
      borderColor: '#0a0f1e',
      borderWidth: 3,
      hoverOffset: 8
    }}]
  }},
  options: {{
    responsive: true,
    cutout: '62%',
    plugins: {{
      legend: {{ position: 'right', labels: {{ color: textColor, font: {{ size: 10 }}, boxWidth: 12, padding: 10 }} }},
      datalabels: {{
        display: true,
        color: '#fff',
        font: {{ size: 9, weight: '700' }},
        formatter: (val, ctx) => {{
          const total = ctx.dataset.data.reduce((a,b)=>a+b,0);
          const pct = (val/total*100).toFixed(1);
          return pct > 4 ? pct+'%' : '';
        }}
      }},
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.label + ': ' + ctx.parsed.toLocaleString() + ' units' }} }}
    }}
  }}
}});

// 5. Price vs Volume Scatter
const scatterData = {json.dumps(scatter_pts)};
new Chart(document.getElementById('scatterChart'), {{
  type: 'bubble',
  data: {{
    datasets: [{{
      label: 'SKU (Price vs Daily Sales)',
      data: scatterData.map(d => ({{ x: d.x, y: d.y, r: d.r }})),
      backgroundColor: scatterData.map((_,i) => PALETTE[i % PALETTE.length] + 'cc'),
      borderColor: scatterData.map((_,i) => PALETTE[i % PALETTE.length]),
      borderWidth: 1
    }}]
  }},
  options: {{
    ...baseOptions,
    plugins: {{
      legend: {{ display: false }},
      datalabels: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: (ctx) => {{
            const d = scatterData[ctx.dataIndex];
            return [' SKU: ' + d.sku, ' Price: $' + d.x, ' Avg Daily: ' + d.y + ' units', ' Total Volume: ' + d.total.toLocaleString()];
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => '$' + v }},
           title: {{ display: true, text: 'Selling Price ($)', color: textColor, font: {{ size: 11 }} }} }},
      y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor }},
           title: {{ display: true, text: 'Avg Daily Units', color: textColor, font: {{ size: 11 }} }} }}
    }}
  }}
}});

// 6. Forecast Bar with Error Bars
const foreLabels = {json.dumps(fore_labels)};
const forePred   = {json.dumps(fore_pred)};
const foreLower  = {json.dumps(fore_lower)};
const foreUpper  = {json.dumps(fore_upper)};

new Chart(document.getElementById('forecastChart'), {{
  type: 'bar',
  data: {{
    labels: foreLabels,
    datasets: [
      {{
        label: 'Lower Bound',
        data: foreLower,
        backgroundColor: 'rgba(239,68,68,0.3)',
        borderRadius: 4,
        stack: 'stack',
        borderSkipped: false
      }},
      {{
        label: 'Predicted Units',
        data: forePred.map((v,i)=>v-foreLower[i]),
        backgroundColor: PALETTE.slice(0,10),
        borderRadius: 4,
        stack: 'stack',
        borderSkipped: false
      }},
      {{
        label: 'Upper Buffer',
        data: foreUpper.map((v,i)=>v-forePred[i]),
        backgroundColor: 'rgba(16,185,129,0.25)',
        borderRadius: 4,
        stack: 'stack',
        borderSkipped: false
      }}
    ]
  }},
  options: {{
    ...baseOptions,
    plugins: {{ ...baseOptions.plugins,
      tooltip: {{
        callbacks: {{
          label: (ctx) => {{
            const i = ctx.dataIndex;
            return [' Predicted: ' + forePred[i].toLocaleString(),
                    ' Lower Bound: ' + foreLower[i].toLocaleString(),
                    ' Upper Bound: ' + foreUpper[i].toLocaleString()];
          }},
          title: (items) => foreLabels[items[0].dataIndex]
        }},
        mode: 'index',
        filter: (item) => item.datasetIndex === 1
      }}
    }},
    scales: {{
      x: {{ stacked: true, grid: {{ color: 'transparent' }}, ticks: {{ color: '#60a5fa', font: {{ size: 9, weight:'600' }}, maxRotation: 35 }} }},
      y: {{ stacked: true, grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toLocaleString() }} }}
    }}
  }}
}});

// 7. Weekly Trend
new Chart(document.getElementById('weeklyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(weekly_labels)},
    datasets: [{{
      label: 'Weekly Units',
      data: {json.dumps(weekly_vals)},
      backgroundColor: 'rgba(139,92,246,0.7)',
      borderColor: '#8b5cf6',
      borderWidth: 1,
      borderRadius: 2
    }}]
  }},
  options: {{
    ...baseOptions,
    plugins: {{ ...baseOptions.plugins, legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: textColor, font: {{ size: 8 }}, maxRotation: 0,
               callback: (val, idx) => idx % 8 === 0 ? {json.dumps(weekly_labels)}[idx]?.slice(0,10) : '' }} }},
      y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toLocaleString() }} }}
    }}
  }}
}});

// 8. Forecast Table
const allForeData = {df_fore.sort_values('predicted unit', ascending=False).to_json(orient='records')};
const tbody = document.getElementById('forecastTableBody');
const maxPred = Math.max(...allForeData.map(d=>d['predicted unit']));
allForeData.forEach((row, i) => {{
  const pct = Math.round(row['predicted unit'] / maxPred * 140);
  const tierName = row['tier_engine'] || 'Production Master Engine';
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="rank">${{i+1}}</td>
    <td class="sku-code">${{row['order sku']}}</td>
    <td style="text-align:right;color:#60a5fa;font-weight:700">${{row['predicted unit'].toLocaleString()}}</td>
    <td style="text-align:right;color:#94a3b8">${{row['lower bound'].toLocaleString()}}</td>
    <td style="text-align:right;color:#10b981">${{row['upper bound'].toLocaleString()}}</td>
    <td style="font-size:10px;color:#cbd5e1">${{tierName}}</td>
    <td><div class="bar-wrap"><div class="mini-bar" style="width:${{pct}}px;
      background:linear-gradient(90deg,${{row['predicted unit']>200?'#3b82f6':'#8b5cf6'}},${{row['predicted unit']>200?'#06b6d4':'#a78bfa'}})"></div>
      <span style="font-size:10px;color:#64748b">${{row['predicted unit']}}</span></div></td>`;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Dashboard successfully generated: {OUT_HTML}")
