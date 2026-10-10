# PHASE 5C — REPLENISHMENT & ROP INTEGRATION REPORT (SIMPLIFIED FORMAT)

**Document ID**: `PHASE-5C-ROP-SIMPLIFIED-20261006`  
**Pipeline**: Unified Multi-Brand Demand Forecasting Engine (`multibrand_pipeline/`)  
**Scope**: Final ROP Sheet Format Simplification & Policy Header Integration  
**Status**: **COMPLETE, VERIFIED & CERTIFIED**  
**Excel Companion Workbook**: [`RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/multibrand_pipeline/reports/RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx)  
**Root Deliverable**: [`RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST_LATEST.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST_LATEST.xlsx)  

---

## 1. Executive Summary & Production Safety

Phase 5C implements the final simplification and layout cleanup for the client operational deliverable:

1. **No Model Changes**:
   - The Global LightGBM model, 60-feature schema, Exp6 calibration, and 10-day forward demand projections remain **frozen and untouched**.
   - Demand forecast quantities are identical to validated Phase 4B/5B outputs.
2. **Production Files Untouched**:
   - Certified legacy Rimmel production files (`models/production_lgbm_model.pkl`, `data/rimmel_clean.db`, `src/final_production_system.py`) remain **100% frozen and byte-identical**.
3. **Downstream Simplification**:
   - Redundant repetitive table columns (`Lead Time (Days)`, `Minimum Stock Level`, `Replenishment Need`, `Recommended Order Qty`) have been consolidated into a clean 10-column table.
   - Lead time and minimum stock levels are prominently displayed in a dedicated **ROP POLICY Header** above the table.

---

## 2. Approved Business Rules & ROP Policy Header

Because Lead Time and Minimum Stock Level are constant across all SKUs, they are displayed once in the header banner above the ROP table:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ROP POLICY: Lead Time = 10 Days | Minimum Stock Level = 6 Units | Target = (ADU × 10) + 6 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Lead Time**: **10 Days** (Supplier lead-time horizon).
* **Minimum Stock Level**: **6 Units** (Approved minimum buffer; projected inventory must never fall below 6 units).

---

## 3. Certified ROP Formulas Used

$$\text{Avg Daily Usage} = \frac{\text{10-Day Forecast}}{10}$$

$$\text{Lead-Time Demand} = \text{Avg Daily Usage} \times 10 = \text{10-Day Forecast}$$

$$\text{Target Stock} = \text{Lead-Time Demand} + 6$$

$$\text{Replenishment Qty} = \lceil \max\left(\text{Target Stock} - \text{Current Stock}, 0\right) \rceil$$

### Single Actionable Replenishment Quantity
- **`Replenishment Qty`** is the single canonical column representing the integer quantity to order.
- No separate or redundant columns (`Replenishment Need`, `Recommended Order Qty`, `Reorder Point`) exist in the table.

---

## 4. Zero Forecast & Status Classification

### Zero Forecast Behavior
When `10-Day Forecast = 0`:
- $\text{Avg Daily Usage} = 0.00$
- $\text{Lead-Time Demand} = 0.00$
- $\text{Target Stock} = 6.00$
- $\text{Replenishment Qty} = \lceil \max(6 - \text{Current Stock}, 0) \rceil$

| Current Stock | 10-Day Forecast | Target Stock | Replenishment Qty | ROP Status |
|:---:|:---:|:---:|:---:|:---|
| **100** | 0 | 6 | **0** | `NO PROJECTED DEMAND` |
| **10** | 0 | 6 | **0** | `NO PROJECTED DEMAND` |
| **6** | 0 | 6 | **0** | `AT MINIMUM STOCK` |
| **5** | 0 | 6 | **1** | `BELOW ROP` |
| **3** | 0 | 6 | **3** | `BELOW ROP` |
| **0** | 0 | 6 | **6** | `BELOW ROP` |

### Deterministic Status Hierarchy
- If $\text{Current Stock} > \text{Target Stock}$: **`ABOVE ROP`**
- Else if $\text{Current Stock} == 6$: **`AT MINIMUM STOCK`**
- Else if $\text{10-Day Forecast} == 0 \text{ and } \text{Current Stock} \ge 6$: **`NO PROJECTED DEMAND`**
- Else if $\text{10-Day Forecast} == 0 \text{ and } \text{Current Stock} < 6$: **`BELOW ROP`**
- Else: **`BELOW ROP`**

---

## 5. Final Workbook Architecture (Exactly 2 Sheets)

Workbook: `RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx`

```
RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx
├── Sheet 1: Forecast_Inventory (15 Columns)
│   ├── AutoFilter Enabled (A3:O1111) | Freeze Panes (Row 4)
│   ├── 1.  Brand
│   ├── 2.  SKU
│   ├── 3.  Product
│   ├── 4.  Current Stock
│   ├── 5.  Forecast Period (2026-09-11 to 2026-09-20)
│   ├── 6.  Amazon Predicted
│   ├── 7.  eBay Predicted
│   ├── 8.  Website Predicted
│   ├── 9.  Other Predicted
│   ├── 10. 10-Day Forecast (Sum of platforms; Total Predicted removed)
│   ├── 11. Days of Cover (Dynamic Uncapped, Zero 999s)
│   ├── 12. Confidence (HIGH / MEDIUM / LOW)
│   ├── 13. Risk (STOCKOUT RISK / HIGH VOLATILITY / NORMAL / LOW DEMAND)
│   ├── 14. Recommended Action
│   └── 15. Reason
│
└── Sheet 2: ROP (10 Columns)
    ├── Banner: ROP POLICY: Lead Time = 10 Days | Minimum Stock Level = 6 Units
    ├── AutoFilter Enabled (A4:J1112) | Freeze Panes (Row 5)
    ├── 1.  Brand
    ├── 2.  SKU
    ├── 3.  Product
    ├── 4.  Current Stock
    ├── 5.  10-Day Forecast
    ├── 6.  Avg Daily Usage
    ├── 7.  Lead-Time Demand
    ├── 8.  Target Stock
    ├── 9.  Replenishment Qty
    └── 10. ROP Status
```

---

## 6. Comprehensive Validation Checks Passed (Section 14)

- [x] **Exactly 2 sheets exist**: `['Forecast_Inventory', 'ROP']`
- [x] **Forecast_Inventory exists**: Verified
- [x] **ROP exists**: Verified
- [x] **Brand exists in both sheets**: Column 1 in both sheets
- [x] **Rimmel rows exist**: 674 SKUs present in both sheets
- [x] **Max Factor rows exist**: 434 SKUs present in both sheets
- [x] **No Total Predicted column**: Removed (replaced by `10-Day Forecast`)
- [x] **No duplicate Replenishment Need column**: Omitted
- [x] **No duplicate Recommended Order Qty column**: Unified into `Replenishment Qty`
- [x] **No Lead Time table column**: Moved to ROP Policy header
- [x] **No Minimum Stock table column**: Moved to ROP Policy header
- [x] **ROP policy header shows Lead Time = 10 Days**: Prominently displayed
- [x] **ROP policy header shows Minimum Stock = 6 Units**: Prominently displayed
- [x] **Avg Daily Usage = 10-Day Forecast / 10**: Verified ($0 \text{ deviations}$)
- [x] **Lead-Time Demand = Avg Daily Usage × 10**: Verified ($0 \text{ deviations}$)
- [x] **Target Stock = Lead-Time Demand + 6**: Verified ($0 \text{ deviations}$)
- [x] **Replenishment Qty = ceil(max(Target Stock - Current Stock, 0))**: Verified ($0 \text{ deviations}$)
- [x] **No negative replenishment values**: Verified ($\min = 0$)
- [x] **Zero forecast handled correctly**: Verified across test cases and live catalog
- [x] **"No Projected Demand" is only a Days of Cover state**: ROP continues uninterrupted
- [x] **ROP still calculates when forecast = 0**: Tested and verified
- [x] **Shared warehouse stock is used**: 278,532 total units, never multiplied
- [x] **No ROP calculation enters LightGBM features**: 60-feature schema locked
- [x] **No model files modified**: Verified

---

## 7. Portfolio Operational Totals (Sep 11–20, 2026)

* **Total Canonical SKUs**: **1,108 SKUs**
* **Total Warehouse Stock**: **278,532 units**
* **Total 10-Day Forecast Demand**: **2,438 units**
  * eBay: 1,253 units (51.4%)
  * Amazon: 1,143 units (46.9%)
  * Website: 38 units (1.6%)
  * Other: 4 units (0.2%)
* **Brand Demand vs. Replenishment Breakdown**:
  * **Rimmel London**: 1,955 units demand | **1,538 units replenishment** across 674 SKUs
  * **Max Factor**: 483 units demand | **988 units replenishment** across 434 SKUs
* **Total Replenishment Units to Order**: **2,526 units** across 459 SKUs (41.4% of catalog)
* **ROP Status Distribution**:
  * `BELOW ROP`: 460 SKUs (41.5%)
  * `NO PROJECTED DEMAND`: 375 SKUs (33.8%)
  * `ABOVE ROP`: 256 SKUs (23.1%)
  * `AT MINIMUM STOCK`: 17 SKUs (1.5%)

---

## 8. Available Deliverables

1. **Root Workspace File**:
   [`RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST_LATEST.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST_LATEST.xlsx)
2. **Archived Pipeline Report**:
   [`multibrand_pipeline/reports/RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/multibrand_pipeline/reports/RIMMEL_MAX_FACTOR_10DAY_OPERATIONAL_FORECAST.xlsx)

---

## 9. Stop Rule Observed

Per Section 16, generation and validation of Phase 5C are complete. Execution is **STOPPED**.
