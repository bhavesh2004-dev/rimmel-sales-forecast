# MYSQL DATABASE SIMPLIFICATION & APPROVED OPERATIONAL FORECAST IMPORT REPORT

**Document ID**: `REPORT-MYSQL-SIMPLIFICATION-20261010-FINAL`  
**Pipeline**: Unified Multi-Brand Demand Forecasting Engine (`multibrand_pipeline/`)  
**Workspace**: `C:\Users\bhave\Desktop\ml_project_multibrand_v2`  
**Database**: MySQL `multibrand_forecasting_dev`  
**Excel Source**: [`MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx`](file:///C:/Users/bhave/Desktop/ml_project_multibrand_v2/artifacts/reports/MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx)  
**Execution Timestamp**: `2026-10-10 15:42:27`  
**Status**: **COMPLETE — AUDITED, BACKED UP, IMPORTED & 100% VERIFIED**

---

## 1. Executive Summary

This report documents the successful database simplification and operational forecast import executed on the `multibrand_forecasting_dev` database. 

In strict adherence to your instructions:
1. **Pre-Simplification Backups Created & Verified**: A full 91.08 MB MySQL logical database dump and individual table-level `.sql` and `.csv` dumps were created for all tables considered for removal.
2. **Approved Operational Forecast Imported**: Exactly **1,433 SKU rows** across all **seven brands** from worksheet `Operational_Forecast_and_ROP` were imported into a newly created canonical table, `operational_forecast_rop`.
3. **Full 21 Business Columns Preserved**: All 21 operational columns were imported without recalculation or silent modifications. Only necessary technical metadata (`run_id`, `brand_id`, `model_id`, `data_cutoff_date`, `created_at`) was appended. No duplicate or redundant forecast quantity columns were created.
4. **Redundant Tables Safely Removed**: After confirming backup integrity and data persistence, 6 redundant output tables (`forecast_results`, `report_forecast_inventory`, `report_rop`, `inventory_results`, `replenishment_results`, and `validation_results`) were safely dropped.
5. **Database Architecture Simplified**: The database now contains **one unified historical sales input** (`normalized_sales`), genuine master/dimension tables, and **exactly one canonical operational forecast output table** (`operational_forecast_rop`).

---

## 2. Pre-Simplification Backup Locations & Manifest

Before any tables were modified or dropped, complete database and table-level backups were exported and verified:

**Backup Directory**:  
`C:\Users\bhave\Desktop\ml_project_multibrand_v2\database_backups\pre_simplification_20261010\`

| Backup Target | Format | Row Count | File Size (Bytes) | Verification Status |
|:---|:---:|:---:|:---:|:---:|
| **Entire Database Dump** | SQL | All Tables | 91,081,286 | Verified Intact |
| `forecast_results` | SQL / CSV | 96,360 | 27,872,479 / 23,270,814 | Verified Intact |
| `report_forecast_inventory` | SQL / CSV | 1,433 | 364,899 / 330,638 | Verified Intact |
| `report_rop` | SQL / CSV | 1,433 | 209,975 / 185,487 | Verified Intact |
| `inventory_results` | SQL / CSV | 1,433 | 282,661 / 253,453 | Verified Intact |
| `replenishment_results` | SQL / CSV | 1,433 | 185,678 / 162,929 | Verified Intact |
| `validation_results` | SQL / CSV | 8 | 3,943 / 1,269 | Verified Intact |

---

## 3. Final MySQL Database Inventory

The simplified schema in `multibrand_forecasting_dev` now comprises exactly **9 active tables**:

```
multibrand_forecasting_dev (9 tables total)
├── Unified Input Sales:
│   └── normalized_sales           134,901 rows (Canonical 7-brand daily transaction grid)
├── Canonical Operational Output:
│   └── operational_forecast_rop     1,433 rows (Sole active operational deliverable)
├── Catalog & Metadata Masters:
│   ├── sku_master                   1,433 rows (Canonical SKU dimension & stock)
│   ├── brand_registry                   7 rows (Brand discovery & coverage limits)
│   ├── pipeline_runs                    6 rows (MLOps pipeline execution registry)
│   ├── model_registry                   1 row  (Model artifact & lineage registry)
│   └── import_batches                   2 rows (Source ingestion batch audit log)
└── Raw Ingestion Staging / Audit:
    ├── raw_rimmel_sales_data      101,085 rows (Rimmel historical raw order lines)
    └── order_sales_data            33,816 rows (Multi-brand historical raw order lines)
```

### Table-by-Table Role & Row Count Summary

| Table Name | Row Count | Architectural Classification | Retention Rationale |
|:---|:---:|:---|:---|
| `normalized_sales` | **134,901** | **Unified Historical Sales Input** | Authoritative normalized daily sales input across all 7 brands. |
| `operational_forecast_rop` | **1,433** | **Canonical Operational Forecast Output** | **Single active operational output table** containing the approved 21 business columns and run metadata. |
| `sku_master` | **1,433** | Master Catalog Dimension | Primary SKU registry containing product titles, categories, current stock, and launch dates. |
| `brand_registry` | **7** | Master Brand Dimension | Brand metadata, status, date coverage, and cutoff boundaries. |
| `pipeline_runs` | **6** | MLOps Registry | Audited record of pipeline executions, statuses, parameters, and time ranges. |
| `model_registry` | **1** | MLOps Registry | Production model versioning, feature lists, and checksum tracking. |
| `import_batches` | **2** | Ingestion Audit | Audit trail of raw data ingestion batches. |
| `raw_rimmel_sales_data` | **101,085** | Raw Ingestion Staging | Immutable raw source staging for Rimmel London. |
| `order_sales_data` | **33,816** | Raw Ingestion Staging | Immutable raw source staging for non-Rimmel brands. |

---

## 4. Removed Redundant Tables

The following 6 redundant output and intermediate tables were dropped from the active database following verified backup:

| Dropped Table Name | Prior Rows | Prior Columns | Rationale for Removal |
|:---|:---:|:---:|:---|
| `forecast_results` | 96,360 | 22 | Redundant daily-grain platform time-series table. Stored multiple historical runs and competing quantity definitions. |
| `report_forecast_inventory` | 1,433 | 18 | Fragmented partial report table (split forecast and basic stock). Superseded by unified 21-column canonical table. |
| `report_rop` | 1,433 | 13 | Fragmented partial ROP table. Superseded by unified 21-column canonical table. |
| `inventory_results` | 1,433 | 12 | Intermediate calculation table from legacy pipeline iterations. |
| `replenishment_results` | 1,433 | 14 | Intermediate calculation table from legacy pipeline iterations. |
| `validation_results` | 8 | 15 | Obsolete historical validation run output. Validation is now documented in standalone audit reports. |

---

## 5. Canonical Table Schema: `operational_forecast_rop`

### DDL Implementation

```sql
CREATE TABLE `operational_forecast_rop` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `run_id` VARCHAR(50) NOT NULL,
    `brand_id` VARCHAR(50) NOT NULL,
    `brand` VARCHAR(100) NOT NULL,
    `sku` VARCHAR(100) NOT NULL,
    `product` VARCHAR(255) NULL,
    `current_stock` INT NOT NULL DEFAULT 0,
    `forecast_period` VARCHAR(50) NOT NULL,
    `amazon_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
    `ebay_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
    `website_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
    `other_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
    `forecast_10d` INT UNSIGNED NOT NULL DEFAULT 0,
    `days_of_cover` VARCHAR(30) NOT NULL,
    `confidence` VARCHAR(20) NOT NULL,
    `risk` VARCHAR(50) NOT NULL,
    `recommended_action` VARCHAR(100) NOT NULL,
    `reason` TEXT NULL,
    `lead_time_days` INT UNSIGNED NOT NULL DEFAULT 10,
    `avg_daily_usage` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    `lead_time_demand` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    `target_stock` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    `replenishment_qty` INT UNSIGNED NOT NULL DEFAULT 0,
    `rop_status` VARCHAR(50) NOT NULL,
    `model_id` VARCHAR(100) NOT NULL DEFAULT 'global_lgbm_model.pkl',
    `data_cutoff_date` DATE NOT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_run_brand_sku` (`run_id`, `brand`, `sku`),
    KEY `idx_run_id` (`run_id`),
    KEY `idx_brand_id` (`brand_id`),
    KEY `idx_sku` (`sku`),
    KEY `idx_rop_status` (`rop_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### Schema Properties
1. **Direct 1:1 Mapping to Excel**: Every one of the 21 columns from the approved workbook is directly represented.
2. **Zero Redundant Quantities**: Only one canonical forecast quantity field (`forecast_10d`) is stored.
3. **Idempotent Unique Key**: `UNIQUE KEY uk_run_brand_sku (run_id, brand, sku)` guarantees that repeated pipeline executions or re-imports update records in place without duplicating rows.
4. **Relational Consistency**: `brand_id` maps cleanly to `brand_registry.brand_id`, and `sku` links to `sku_master.canonical_sku`.

---

## 6. Verification Results

A series of automated SQL assertions and consistency checks were executed post-import:

### 6.1 Row Count & Brand Distribution

| Brand Name | Brand ID | Expected SKUs | Imported MySQL Rows | Data Cutoff Date ($T$) | Forward Forecast Horizon | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rimmel** | `RIMMEL` | 767 | **767** | `2026-09-10` | `2026-09-11` to `2026-09-20` | **MATCH (100%)** |
| **Max Factor** | `MAX_FACTOR` | 475 | **475** | `2026-10-06` | `2026-10-07` to `2026-10-16` | **MATCH (100%)** |
| **Kifra** | `KIFRA` | 68 | **68** | `2026-10-06` | `2026-10-07` to `2026-10-16` | **MATCH (100%)** |
| **Weleda** | `WELEDA` | 61 | **61** | `2026-10-05` | `2026-10-06` to `2026-10-15` | **MATCH (100%)** |
| **delilah** | `DELILAH` | 29 | **29** | `2026-10-02` | `2026-10-03` to `2026-10-12` | **MATCH (100%)** |
| **Geek & Gorgeous** | `GEEK_GORGEOUS` | 20 | **20** | `2026-10-06` | `2026-10-07` to `2026-10-16` | **MATCH (100%)** |
| **Frank Body** | `FRANK_BODY` | 13 | **13** | `2026-10-02` | `2026-10-03` to `2026-10-12` | **MATCH (100%)** |
| **TOTAL** | — | **1,433** | **1,433** | — | — | **MATCH (100%)** |

### 6.2 Channel Volumetric Integrity (Additive Channel Law)
$$\sum \text{Amazon} + \sum \text{eBay} + \sum \text{Website} + \sum \text{Other} \equiv \sum \text{Forecast 10D}$$
- **Amazon Predicted**: 1,104 units
- **eBay Predicted**: 1,486 units
- **Website Predicted**: 285 units
- **Other Predicted**: 172 units
- **Total Calculated Channels**: **3,047 units**
- **Canonical `forecast_10d` Total**: **3,047 units**
- **Discrepancy / Violations**: **0 units across all 1,433 SKUs**.

### 6.3 Replenishment & Stock Buffer Verification
$$\text{Current Stock} + \text{Replenishment Qty} - \text{Lead-Time Demand} \ge 6.00 \text{ units}$$
- **Total Current Warehouse Stock**: 280,058 units
- **Total Replenishment Units Needed**: 4,101 units
- **SKUs Requiring Purchase Order**: 737 SKUs (51.4%)
- **Minimum Observed Projected Stock Buffer**: **6.00 units** (0 violations)
- **Negative Replenishment Quantities**: **0**
- **Sentinel Placeholders (`999`)**: **0**

### 6.4 Idempotency & Repeat Import Check
The import routine was executed twice in succession:
- First execution: Inserted 1,433 rows.
- Second execution: Updated 1,433 rows in place via `ON DUPLICATE KEY UPDATE`.
- Final table row count: **1,433 rows** (zero phantom duplicates or row inflation).

---

## 7. Deliverable Paths & Production Integrity Summary

| Resource | Path | Verification Details |
|:---|:---|:---|
| **Canonical MySQL Table** | `multibrand_forecasting_dev.operational_forecast_rop` | 1,433 rows, 21 business columns + run metadata |
| **Unified Sales Table** | `multibrand_forecasting_dev.normalized_sales` | 134,901 rows (100% untouched) |
| **Approved Excel Deliverable** | [`MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx`](file:///C:/Users/bhave/Desktop/ml_project_multibrand_v2/artifacts/reports/MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx) | Certified 21-column operational workbook |
| **Pipeline Runner Script** | [`simplify_and_import_operational_forecast.py`](file:///C:/Users/bhave/Desktop/ml_project_multibrand_v2/multibrand_pipeline/src/simplify_and_import_operational_forecast.py) | Automated, idempotent import & verification utility |
| **Database Backups** | `C:\Users\bhave\Desktop\ml_project_multibrand_v2\database_backups\pre_simplification_20261010\` | Full 91MB logical dump + table-level SQL and CSV exports |
| **Active Production Model** | `multibrand_pipeline/models/global_lgbm_model.pkl` | MD5: `dac42213d9c8c3a63f97e6171bf187d7` (Untouched) |
| **Preserved Legacy Model** | `multibrand_pipeline/models/backup_rimmel_maxfactor_lgbm_model.pkl` | MD5: `db098afc3d8f1805dd7f21d2a193f0ca` (Untouched) |

---
**Certification**: The database is now simplified to one unified input (`normalized_sales`) and one canonical operational output (`operational_forecast_rop`), fully validated against all business constraints and ready for enterprise consumption.
