"""
Verification and Sample Generation Script for Real Jev Phase 1 Experiment
========================================================================
Extracts 28 representative SKU x Platform cases from SQLite data/rimmel_clean.db
as of decision date 2026-09-10 with zero future data leakage.
"""
import sqlite3
import pandas as pd
import numpy as np

def extract_sample_cases():
    conn = sqlite3.connect('data/rimmel_clean.db')
    query = """
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
    WHERE date = '2026-09-10'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Derived Days of Cover (DoC) based strictly on historical v14 run rate
    df['days_of_cover'] = df.apply(
        lambda r: round(r['current_stock'] / r['v14'], 1) if r['v14'] > 0 else (999.0 if r['current_stock'] > 0 else 0.0),
        axis=1
    )

    # 8 Mutually Exclusive Sampling Strata with fixed seed 42
    s1 = df[(df['v14'] >= 1.0) & (df['current_stock'] < 30)].copy()
    s1['stratum'] = '1. High Velocity + Low Stock'

    s2_cand = df[(df['v14'] >= 1.0) & (df['current_stock'] >= 100)]
    s2 = s2_cand.sample(n=4, random_state=42).copy()
    s2['stratum'] = '2. High Velocity + Healthy Stock'

    s3_cand = df[(df['v14'] >= 0.2) & (df['v14'] < 1.0) & (df['current_stock'] < 20)]
    s3 = s3_cand.sample(n=4, random_state=42).copy()
    s3['stratum'] = '3. Medium Velocity + Low Stock'

    s4_cand = df[(df['v14'] >= 0.2) & (df['v14'] < 1.0) & (df['current_stock'] >= 50)]
    s4 = s4_cand.sample(n=4, random_state=42).copy()
    s4['stratum'] = '4. Medium Velocity + Healthy Stock'

    s5_cand = df[(df['v14'] < 0.1) & (df['current_stock'] == 0)]
    s5 = s5_cand.sample(n=3, random_state=42).copy()
    s5['stratum'] = '5. Low/Zero Velocity + Stockout'

    s6_cand = df[(df['v14'] < 0.1) & (df['current_stock'] > 100)]
    s6 = s6_cand.sample(n=3, random_state=42).copy()
    s6['stratum'] = '6. Low Velocity + High Stock'

    prev_keys = set(pd.concat([s1, s2, s3, s4, s5, s6])[['canonical_sku', 'platform_group']].itertuples(index=False, name=None))

    s7_cand = df[((df['ebay_promoted_flag'] == 1) | (df['amazon_sessions_momentum'] > 1.2)) & 
                 (~df.set_index(['canonical_sku', 'platform_group']).index.isin(prev_keys))]
    s7 = s7_cand.sample(n=3, random_state=42).copy()
    s7['stratum'] = '7. Promotion / Traffic Momentum'

    prev_keys.update(s7[['canonical_sku', 'platform_group']].itertuples(index=False, name=None))

    s8_cand = df[(df['cv_30'] > 1.2) & (df['v14'] > 0.2) & 
                 (~df.set_index(['canonical_sku', 'platform_group']).index.isin(prev_keys))]
    s8 = s8_cand.sample(n=3, random_state=42).copy()
    s8['stratum'] = '8. High Volatility / Erratic'

    sample_df = pd.concat([s1, s2, s3, s4, s5, s6, s7, s8]).reset_index(drop=True)
    return sample_df

if __name__ == '__main__':
    df = extract_sample_cases()
    print(f"Total sampled cases: {len(df)}")
    for idx, r in df.iterrows():
        print(f"Case {idx+1:02d}: {r['canonical_sku']} [{r['platform_group']}] - {r['category']} | Stk: {r['current_stock']} | v14: {r['v14']:.2f} | DoC: {r['days_of_cover']}d | {r['stratum']}")
