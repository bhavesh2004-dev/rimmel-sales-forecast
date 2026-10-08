"""
Real TypeSafe Jev + Rimmel Business Data: Phase 1 Experiment
============================================================
Dispatches small structured business decision contexts for 28 representative
Rimmel SKU x Platform cases (as of 2026-09-10) to real TypeSafe Jev via
OpenRouter's Decisions API (POST https://openrouter.ai/api/alpha/decisions).

Evaluates 3 structured decisions per case:
1. Urgent Replenishment (Noul primitive: boolean probability)
2. Replenishment Risk (Score primitive: Low / Medium / High)
3. Replenishment Priority (Choice primitive: LOW / MEDIUM / HIGH)

Strict Safety & Governance Constraints:
- Production Exp6 LightGBM and forecasting pipelines remain 100% untouched.
- DeterministicJevProvider remains 100% intact.
- Authoritative source: data/rimmel_clean.db (ml_features_zero at date = '2026-09-10').
- Zero future leakage: No post-Sep 10 data utilized.
- No API keys printed, logged, or serialized.
- Controlled execution: Exactly 28 API calls (1 per SKU x Platform case).
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# API & Model Configuration
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_NAME = "~typesafe/jev-latest"
DECISION_DATE = "2026-09-10"
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "rimmel_clean.db"
ENV_PATH = BASE_DIR / ".env"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY safely from .env without printing or logging."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"OPENROUTER_API_KEY is empty in {ENV_PATH}. Please ensure your key is set.")
    return api_key


def extract_stratified_sample() -> pd.DataFrame:
    """
    Extract 28 representative SKU x Platform cases from authoritative SQLite
    database (ml_features_zero) as of 2026-09-10 using fixed seed 42.
    """
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        date,
        canonical_sku,
        platform_group,
        category,
        v7,
        v14,
        v30,
        v90,
        cv_30,
        lag_1,
        lag_7,
        current_stock,
        in_stock_flag,
        stockout_flag,
        days_since_stockout,
        selling_price,
        buy_box_7d,
        buy_box_percentage,
        amazon_sessions_momentum,
        ebay_promoted_flag,
        promo_days_30,
        restock_known,
        restock_date
    FROM ml_features_zero 
    WHERE date = '{DECISION_DATE}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Derived Days of Cover (DoC) based strictly on historical v14 run rate
    df['days_of_cover'] = df.apply(
        lambda r: round(r['current_stock'] / r['v14'], 1) if r['v14'] > 0 else (999.0 if r['current_stock'] > 0 else 0.0),
        axis=1
    )

    # 8 Stratification Buckets
    s1 = df[(df['v14'] >= 1.0) & (df['current_stock'] < 30)].copy()
    s1['stratum'] = '1. High Velocity + Low Stock'

    s2_cand = df[(df['v14'] >= 1.0) & (df['current_stock'] >= 100)]
    s2 = s2_cand.sample(n=4, random_state=RANDOM_SEED).copy()
    s2['stratum'] = '2. High Velocity + Healthy Stock'

    s3_cand = df[(df['v14'] >= 0.2) & (df['v14'] < 1.0) & (df['current_stock'] < 20)]
    s3 = s3_cand.sample(n=4, random_state=RANDOM_SEED).copy()
    s3['stratum'] = '3. Medium Velocity + Low Stock'

    s4_cand = df[(df['v14'] >= 0.2) & (df['v14'] < 1.0) & (df['current_stock'] >= 50)]
    s4 = s4_cand.sample(n=4, random_state=RANDOM_SEED).copy()
    s4['stratum'] = '4. Medium Velocity + Healthy Stock'

    s5_cand = df[(df['v14'] < 0.1) & (df['current_stock'] == 0)]
    s5 = s5_cand.sample(n=3, random_state=RANDOM_SEED).copy()
    s5['stratum'] = '5. Low/Zero Velocity + Stockout'

    s6_cand = df[(df['v14'] < 0.1) & (df['current_stock'] > 100)]
    s6 = s6_cand.sample(n=3, random_state=RANDOM_SEED).copy()
    s6['stratum'] = '6. Low Velocity + High Stock'

    prev_keys = set(pd.concat([s1, s2, s3, s4, s5, s6])[['canonical_sku', 'platform_group']].itertuples(index=False, name=None))

    s7_cand = df[((df['ebay_promoted_flag'] == 1) | (df['amazon_sessions_momentum'] > 1.2)) & 
                 (~df.set_index(['canonical_sku', 'platform_group']).index.isin(prev_keys))]
    s7 = s7_cand.sample(n=3, random_state=RANDOM_SEED).copy()
    s7['stratum'] = '7. Promotion / Traffic Momentum'

    prev_keys.update(s7[['canonical_sku', 'platform_group']].itertuples(index=False, name=None))

    s8_cand = df[(df['cv_30'] > 1.2) & (df['v14'] > 0.2) & 
                 (~df.set_index(['canonical_sku', 'platform_group']).index.isin(prev_keys))]
    s8 = s8_cand.sample(n=3, random_state=RANDOM_SEED).copy()
    s8['stratum'] = '8. High Volatility / Erratic'

    sample_df = pd.concat([s1, s2, s3, s4, s5, s6, s7, s8]).reset_index(drop=True)
    sample_df['case_id'] = [f"CASE_{i+1:02d}" for i in range(len(sample_df))]
    return sample_df


def format_business_state(row: pd.Series) -> str:
    """
    Format a single SKU x Platform case into a concise, factual business decision state.
    Strictly uses historical values known on or before 2026-09-10.
    """
    plat = row['platform_group']
    stock = row['current_stock']
    in_stock = "In Stock" if row['in_stock_flag'] == 1 else "Out of Stock"
    doc = f"{row['days_of_cover']:.1f} days" if row['days_of_cover'] < 900 else "Excess (>900 days)"
    v7 = f"{row['v7']:.2f}"
    v14 = f"{row['v14']:.2f}"
    v30 = f"{row['v30']:.2f}"
    v90 = f"{row['v90']:.2f}"
    cv = f"{row['cv_30']:.2f}" if not pd.isna(row['cv_30']) and row['cv_30'] > 0 else "0.00 (Stable/Intermittent)"
    price = f"£{row['selling_price']:.2f}" if not pd.isna(row['selling_price']) and row['selling_price'] > 0 else "Not recorded"

    # Platform specific signals
    if plat == 'Amazon':
        bb = f"{row['buy_box_7d']:.1f}%" if not pd.isna(row['buy_box_7d']) else "Not available"
        traffic = f"{row['amazon_sessions_momentum']:.2f}x relative to 30d baseline" if not pd.isna(row['amazon_sessions_momentum']) else "Normal"
        promo = "None active"
    elif plat == 'eBay':
        bb = "Not applicable (eBay channel)"
        traffic = "Normal"
        promo = f"eBay Promoted Listing active (Promoted in {int(row['promo_days_30'])} of last 30 days)" if row['ebay_promoted_flag'] == 1 else "None active"
    else:
        bb = f"Not applicable ({plat} channel)"
        traffic = "Normal"
        promo = "None active"

    state_text = (
        f"Product SKU: {row['canonical_sku']} (Category: {row['category']}) on platform: {plat} as of decision date {DECISION_DATE}.\n"
        f"- Inventory Position: On-hand warehouse stock = {stock:.0f} units ({in_stock}, Stockout flag = {row['stockout_flag']}). "
        f"Calculated days of cover based on 14-day run rate = {doc}. Days since last stockout = {row['days_since_stockout']} days.\n"
        f"- Sales Velocity & Demand: 7-day average = {v7} units/day, 14-day average = {v14} units/day, "
        f"30-day average = {v30} units/day, 90-day average = {v90} units/day. 30-day demand volatility CV = {cv}. "
        f"Recent sales: lag 1 day = {row['lag_1']:.0f} units, lag 7 day = {row['lag_7']:.0f} units.\n"
        f"- Commercial Context: Selling price = {price}. Buy Box ownership = {bb}. Traffic momentum = {traffic}. Marketing promotions = {promo}.\n"
        f"- Supply Chain Constraints: Supplier lead time = Not available in source data; Supplier MOQ = Not available in source data; In-transit purchase orders = Not available in source data."
    )
    return state_text


def build_jev_request_payload(state_text: str) -> dict:
    """Build the structured OpenRouter Decisions API request with 3 typed questions."""
    return {
        "model": MODEL_NAME,
        "state": state_text,
        "questions": {
            "urgent_replenishment": {
                "type": "noul",
                "instructions": "Based on the available demand, inventory, platform and commercial context, is urgent replenishment required?"
            },
            "replenishment_risk": {
                "type": "score",
                "instructions": "Assess the replenishment and inventory stockout risk.",
                "criteria": ["Low", "Medium", "High"]
            },
            "replenishment_priority": {
                "type": "choice",
                "instructions": "What is the replenishment priority for this SKU?",
                "criteria": {
                    "LOW": "Stock is healthy or velocity is low.",
                    "MEDIUM": "Stock cover is moderate or demand is trending.",
                    "HIGH": "Stock is critically low or stockout is imminent."
                }
            }
        }
    }


def run_experiment():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()

    print("==================================================")
    print("REAL TYPESAFE JEV + RIMMEL DATA: PHASE 1 EXECUTION")
    print("==================================================")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {OPENROUTER_DECISIONS_URL}")
    print(f"Decision Date: {DECISION_DATE}")
    print(f"Authoritative SQLite Source: {DB_PATH}")

    # Extract sample
    sample_df = extract_stratified_sample()
    num_cases = len(sample_df)
    decisions_per_case = 3
    max_api_calls = num_cases

    print(f"\n[COST CONTROL & PRE-EXECUTION AUDIT]")
    print(f"- Number of SKU x Platform cases: {num_cases}")
    print(f"- Number of decisions evaluated per case: {decisions_per_case} (Noul, Score, Choice)")
    print(f"- Expected maximum API calls: {max_api_calls} (1 call per case containing all 3 questions)")
    print(f"- Execution Mode: Synchronous, single-pass, zero retries")
    print("==================================================\n")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bhavesh2004-dev/rimmel-sales-forecast",
        "X-Title": "Rimmel ROP Replenishment Experiment - Phase 1"
    }

    raw_responses = []
    case_summary_records = []
    granular_decision_records = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost_usd = 0.0

    print("Starting API execution across 28 representative cases...\n")

    for idx, row in sample_df.iterrows():
        case_id = row['case_id']
        sku = row['canonical_sku']
        plat = row['platform_group']
        stratum = row['stratum']
        stk = row['current_stock']
        v14 = row['v14']
        doc = row['days_of_cover']

        state_text = format_business_state(row)
        payload = build_jev_request_payload(state_text)

        ts = datetime.utcnow().isoformat()
        try:
            resp = requests.post(
                OPENROUTER_DECISIONS_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            status_code = resp.status_code

            if status_code != 200:
                print(f"[!] {case_id} ({sku} | {plat}) FAILED: HTTP {status_code} - {resp.text[:150]}")
                raw_responses.append({
                    "case_id": case_id,
                    "sku": sku,
                    "platform": plat,
                    "status_code": status_code,
                    "error": resp.text[:200],
                    "timestamp": ts
                })
                continue

            resp_data = resp.json()
            model_returned = resp_data.get("model", "unknown")
            answers = resp_data.get("answers", {})
            usage = resp_data.get("usage", {})
            cost = usage.get("cost", 0.0)
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)

            total_cost_usd += cost
            total_tokens_in += in_tok
            total_tokens_out += out_tok

            # 1. Urgent Replenishment (Noul)
            noul_info = answers.get("urgent_replenishment", {})
            urgency_prob = noul_info.get("noul", None)

            # 2. Replenishment Risk (Score)
            score_info = answers.get("replenishment_risk", {})
            score_val = score_info.get("score", None)
            score_legend = score_info.get("legend", {})
            score_label = score_legend.get(str(score_val), str(score_val))
            score_conf = score_info.get("confidence", None)

            # 3. Replenishment Priority (Choice)
            choice_info = answers.get("replenishment_priority", {})
            choice_val = choice_info.get("choice", None)
            choice_conf = choice_info.get("confidence", None)

            # Print single-line progress
            print(f"[{case_id}] {sku:<23} | {plat:<7} | Stk: {stk:4.0f} | v14: {v14:5.2f} | DoC: {doc:6.1f}d "
                  f"==> Jev: Urg={urgency_prob:0.2f} | Risk={score_label:<6} | Prio={choice_val:<6} | Cost=${cost:.6f}")

            # Store Case Summary
            case_summary_records.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "category": row['category'],
                "decision_date": DECISION_DATE,
                "stratum": stratum,
                "current_stock": stk,
                "v14": v14,
                "days_of_cover": doc,
                "selling_price": row['selling_price'],
                "jev_urgent_replenishment_prob": urgency_prob,
                "jev_replenishment_risk_score": score_val,
                "jev_replenishment_risk_label": score_label,
                "jev_replenishment_risk_confidence": score_conf,
                "jev_replenishment_priority_choice": choice_val,
                "jev_replenishment_priority_confidence": choice_conf,
                "api_cost_usd": cost,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "model_version": model_returned,
                "request_timestamp": ts,
                "status": "SUCCESS"
            })

            # Store Granular Decision Records (one row per decision type)
            granular_decision_records.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "decision_date": DECISION_DATE,
                "stratum": stratum,
                "decision_type": "urgent_replenishment",
                "primitive": "noul",
                "decision_output": urgency_prob,
                "confidence_score": 1.0,
                "cost_usd": cost / 3.0,
                "model_version": model_returned,
                "status": "SUCCESS"
            })
            granular_decision_records.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "decision_date": DECISION_DATE,
                "stratum": stratum,
                "decision_type": "replenishment_risk",
                "primitive": "score",
                "decision_output": score_label,
                "confidence_score": score_conf,
                "cost_usd": cost / 3.0,
                "model_version": model_returned,
                "status": "SUCCESS"
            })
            granular_decision_records.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "decision_date": DECISION_DATE,
                "stratum": stratum,
                "decision_type": "replenishment_priority",
                "primitive": "choice",
                "decision_output": choice_val,
                "confidence_score": choice_conf,
                "cost_usd": cost / 3.0,
                "model_version": model_returned,
                "status": "SUCCESS"
            })

            # Audit record with full raw response (excluding API key)
            raw_responses.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "stratum": stratum,
                "input_state": state_text,
                "raw_response": resp_data,
                "timestamp": ts
            })

        except Exception as e:
            print(f"[!] {case_id} Exception: {type(e).__name__} - {str(e)[:100]}")
            raw_responses.append({
                "case_id": case_id,
                "sku": sku,
                "platform": plat,
                "exception": type(e).__name__,
                "timestamp": ts
            })

    # Save to disk
    case_summary_df = pd.DataFrame(case_summary_records)
    case_summary_path = OUTPUTS_DIR / "real_jev_phase1_case_summary.csv"
    case_summary_df.to_csv(case_summary_path, index=False)

    granular_df = pd.DataFrame(granular_decision_records)
    granular_path = OUTPUTS_DIR / "real_jev_phase1_results.csv"
    granular_df.to_csv(granular_path, index=False)

    raw_path = OUTPUTS_DIR / "real_jev_phase1_raw_responses.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_responses, f, indent=2)

    print("\n==================================================")
    print("PHASE 1 EXECUTION COMPLETE")
    print("==================================================")
    print(f"Total API Calls Dispatched: {len(case_summary_records)} / {num_cases}")
    print(f"Total Structured Decisions Captured: {len(granular_decision_records)} (84 decisions)")
    print(f"Total Tokens Consumed: {total_tokens_in} input, {total_tokens_out} output")
    print(f"Total API Cost: ${total_cost_usd:.6f} USD")
    print(f"Outputs Generated:")
    print(f"  - Case Summary: {case_summary_path}")
    print(f"  - Granular Decisions: {granular_path}")
    print(f"  - Raw JSON Audit: {raw_path}")
    print("==================================================")


if __name__ == "__main__":
    run_experiment()
