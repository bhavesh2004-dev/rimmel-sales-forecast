# STREAMLIT CLOUD SERVERLESS DEPLOYMENT GUIDE (NO BACKEND REQUIRED)

**Project:** Multi-Brand Demand Forecasting & Inventory Planning Platform  
**Target Platform:** [Streamlit Community Cloud](https://share.streamlit.io/)  
**Live URL:** [https://rimmel-sales-forecast.streamlit.app](https://rimmel-sales-forecast.streamlit.app)  
**Main Entrypoint:** `app.py`  
**Architecture:** Serverless Embedded Architecture (Direct SQLite + Parquet)

---

## 1. THE BIG ANSWER: CAN WE DEPLOY WITHOUT FASTAPI?

**YES, 100% ABSOLUTELY.**

You do **NOT** need to deploy FastAPI, uvicorn, or any separate backend server to Streamlit Cloud. 

The application (`app.py`) is designed as a **Self-Contained Embedded System**. It runs autonomously on Streamlit Community Cloud using Python's built-in SQLite engine and compact serialized datasets bundled directly inside your GitHub repository.

---

## 2. HOW OUR ARCHITECTURE WORKS

### The Misconception: "Does Streamlit need a running FastAPI backend?"
No. Many web developers assume that because there is a `backend/` directory in the project, Streamlit must make HTTP calls (`fetch()` or `requests.get("http://localhost:8000/...")`) to a running web server.

In this platform:
- The **FastAPI backend** was built as an optional REST API for third-party microservices (or the legacy Next.js UI).
- The **Streamlit dashboard (`app.py`) does NOT make HTTP API calls**. Instead, it accesses the production database **directly in-memory and on-disk**.

### Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                               OFFLINE / LOCAL ENVIRONMENT (BATCH PIPELINE)                        |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   Raw Multi-Brand Sales Data (Excel/CSV)                                                         |
|                     │                                                                             |
|                     ▼                                                                             |
|   Offline ML Forecasting Pipeline (LightGBM Exp6 + Zero Treatment + ROP Safety Guard)             |
|                     │                                                                             |
|                     ▼                                                                             |
|   Embedded Operational Artifacts:                                                                |
|     1. data/app_multibrand.db (SQLite database — exactly 2.2 MB)                                  |
|     2. data/order_sales_data.csv (Max Factor transactions — 7.2 MB)                               |
|     3. data/processed/dashboard_historical_daily.parquet (Rimmel actuals — 1.0 MB)               |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
                                              │
                                 [git push origin main to GitHub]
                                              │
                                              ▼
+---------------------------------------------------------------------------------------------------+
|                                 STREAMLIT COMMUNITY CLOUD (PRODUCTION)                            |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   Single Streamlit Worker Container:                                                              |
|     • Installs: requirements.txt (streamlit, pandas, plotly, openpyxl, pyarrow)                   |
|     • Executes: streamlit run app.py                                                              |
|                                                                                                   |
|     app.py (Embedded Zero-Latency Engine)                                                         |
|         │                                                                                         |
|         ├─── Reads data/app_multibrand.db via built-in sqlite3 (Read-Only)                        |
|         ├─── Reads data/order_sales_data.csv for Max Factor history                               |
|         └─── Reads data/processed/*.parquet for Rimmel history                                    |
|                                                                                                   |
|     User Browser (Planners / Supply Chain Executives)                                             |
|     URL: https://rimmel-sales-forecast.streamlit.app                                              |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. WHY SQLITE IS PERFECT FOR STREAMLIT CLOUD

1. **Serverless & Zero Maintenance:**  
   Unlike PostgreSQL, MySQL, or SQL Server, **SQLite does not run as a server daemon**. It is an embedded C library built right into the Python standard library (`import sqlite3`). It reads directly from a `.db` file on disk.
2. **Compact File Size:**  
   Our entire multi-brand production catalog (`data/app_multibrand.db`) containing all 1,108 SKUs, platform forecasts, inventory statuses, and replenishment orders is only **2.2 Megabytes**. GitHub easily allows files up to 100 MB.
3. **Blazing Fast Performance (<50 milliseconds):**  
   Because the file is stored locally in the container's virtual disk, there is **zero network latency** and no HTTP round-trip delays. Combined with `@st.cache_data`, pages and tabs render instantly.
4. **Cost:**  
   **$0.00 / completely free.** You don't need AWS EC2, Heroku dynos, Docker containers, or Render backend instances.

---

## 4. WHAT FILES ARE REQUIRED FOR STREAMLIT CLOUD

When Streamlit Cloud clones your repository, it only needs these essential files:

| File / Folder | Size | Role in Production |
|---|---|---|
| `app.py` | ~56 KB | Main Streamlit application entrypoint (UI, charts, 6 tabs, filters) |
| `.streamlit/config.toml` | <1 KB | Configures Dark Theme, port, and headless server settings |
| `requirements.txt` | <1 KB | Python dependencies (streamlit, pandas, plotly, openpyxl, pyarrow) |
| `data/app_multibrand.db` | **2.2 MB** | Master production database for run `RUN-REAL-PROD-2026` |
| `data/order_sales_data.csv` | **7.2 MB** | Authentic transaction history for Max Factor SKUs |
| `data/processed/dashboard_historical_daily.parquet` | **1.0 MB** | Authentic daily transaction actuals for Rimmel SKUs |
| `data/processed/dashboard_validation_sku_daily.csv` | ~450 KB | Holdout validation records (Sep 01–10, 2026) |
| `data/processed/dashboard_forecast_sku_daily.csv` | ~380 KB | Forward daily forecast records (Sep 11–20, 2026) |
| `reports/validation_metrics.csv` | <2 KB | Metric report by selling channel |

*(Note: Large raw training databases like `data/rimmel_clean.db` (694 MB) are **NOT needed** on Streamlit Cloud because their aggregated features were already compiled into the compact `.parquet` and `.db` files).*

---

## 5. STEP-BY-STEP DEPLOYMENT PROCESS

Follow these exact steps to deploy to Streamlit Community Cloud:

### Step 1: Stage and Commit the Files to Git
Open PowerShell in `C:\Users\bhave\Desktop\ml_project` and run:

```powershell
# 1. Stage the application, configuration, and embedded production datasets
git add app.py .streamlit/config.toml .gitignore requirements.txt
git add data/app_multibrand.db data/order_sales_data.csv

# 2. Commit the changes
git commit -m "feat: deploy standalone multi-brand dashboard with embedded sqlite database"

# 3. Push to GitHub
git push origin main
```

---

### Step 2: Open Streamlit Community Cloud
1. Go to your web browser and open:  
   **[https://share.streamlit.io/](https://share.streamlit.io/)**
2. Sign in with your GitHub account (`bhavesh2004-dev`).

---

### Step 3: Configure Your App Settings
Click **"New app"** (or find your existing app `rimmel-sales-forecast` and click **"Reboot / Settings"**):

* **Repository:** `bhavesh2004-dev/rimmel-sales-forecast`
* **Branch:** `main`
* **Main file path:** `app.py`
* **App URL:** `https://rimmel-sales-forecast.streamlit.app` (or any custom subdomain)

---

### Step 4: Click "Deploy!"
Streamlit Cloud will:
1. Provision a lightweight Linux container.
2. Clone your `main` branch.
3. Automatically install all packages from `requirements.txt`.
4. Launch `streamlit run app.py`.
5. Open your live multi-brand dashboard in ~60 seconds.

---

## 6. HOW DATA UPDATES WILL WORK IN THE FUTURE

When business operations change or when you want to update the forecast in the future:

1. **Option A: Drag-and-Drop Ingestion (In-App)**
   - Planners can use the built-in uploader expander: **"📤 Upload Multi-Brand Dataset (.xlsx, .xls, .csv)"** to inspect new catalog uploads directly in their browser.
2. **Option B: Push a New Production Run (Offline)**
   - When you retrain models locally with fresh sales data, the pipeline writes the updated results into `data/app_multibrand.db`.
   - You simply run:
     ```powershell
     git add data/app_multibrand.db
     git commit -m "data: update operational forecast run"
     git push origin main
     ```
   - Streamlit Cloud automatically detects the git push and reloads the live dashboard with the new forecast in less than 30 seconds!

---

## 7. SUMMARY CHECKLIST

- [x] **No backend server required:** FastAPI does not need to be hosted or run.
- [x] **Zero external API dependencies:** Direct local SQLite & Parquet reads.
- [x] **File size compliant:** All tracked data files are $<10$ MB (GitHub limit is 100 MB).
- [x] **Dark theme configured:** `.streamlit/config.toml` pre-configured.
- [x] **Multi-brand ready:** All Brands (1,108 SKUs), Rimmel (674 SKUs), Max Factor (434 SKUs).
- [x] **Dual reports included:** Current View CSV and Multi-sheet Full Brand Excel workbook export.
