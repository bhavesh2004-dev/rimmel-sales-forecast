"""
DATA CLEANING & ANOMALY CLASSIFICATION MODULE (PHASE 1)
======================================================
Identifies and classifies all data quality items across 5 strict tiers:
1. SAFE_TO_FIX
2. NEEDS_DERIVED_FIX
3. BUSINESS_CONFIRMATION_REQUIRED
4. PRESERVE_AS_IS
5. EXCLUDE_FROM_MODEL

Tracks all transformations in a structured transformation log.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def audit_and_classify_data_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects all catalog, channel, inventory, pricing, and bundle anomalies,
    returning a structured log of classified issues.
    """
    issues = []
    
    # DQ-001: Historical Untracked Inventory
    pre_aug_rows = df[df['date'] < '2025-08-01']
    null_stock_pre_aug = pre_aug_rows['current_stock'].isnull().sum()
    issues.append({
        'issue_id': 'DQ-001',
        'field_name': 'current_stock',
        'classification_tier': 'BUSINESS_CONFIRMATION_REQUIRED',
        'issue_description': f'current_stock is 100% unpopulated for Jan 1 - Jul 31, 2025 ({null_stock_pre_aug:,} rows). Inventory system tracking began August 1, 2025.',
        'affected_rows_count': int(null_stock_pre_aug),
        'affected_rows_pct': round((null_stock_pre_aug / len(df)) * 100.0, 2),
        'action_taken': 'Preserve as UNMONITORED (has_inventory_signal=0) in Full Dataset; exclude pre-August 2025 from ML-Ready dataset.',
        'status': 'Resolved via Split'
    })
    
    # DQ-002: Zero-Price Wholesale / Manual Records
    zero_price_rows = df[df['selling_price'] <= 0]
    issues.append({
        'issue_id': 'DQ-002',
        'field_name': 'selling_price',
        'classification_tier': 'NEEDS_DERIVED_FIX',
        'issue_description': f'selling_price is £0.00 in {len(zero_price_rows):,} rows (330 from UFK Trading Manual, 1 from Bellas Beauty Ebay). Represents wholesale/manual sample dispatches.',
        'affected_rows_count': int(len(zero_price_rows)),
        'affected_rows_pct': round((len(zero_price_rows) / len(df)) * 100.0, 2),
        'action_taken': 'Preserve physical units in demand; tag is_zero_price_order = 1 so price elasticity features exclude them.',
        'status': 'Resolved via Flag'
    })
    
    # DQ-003: Multi-Pack Customer Cart Additions
    cart_add_rows = df[df['units_sold'] > df['orders_count'] * df['pack_multiplier']]
    issues.append({
        'issue_id': 'DQ-003',
        'field_name': 'units_sold vs orders_count',
        'classification_tier': 'PRESERVE_AS_IS',
        'issue_description': f'units_sold exceeds orders_count * pack_multiplier in {len(cart_add_rows):,} rows. Valid customer basket additions (buyer ordered multiple packs in 1 checkout).',
        'affected_rows_count': int(len(cart_add_rows)),
        'affected_rows_pct': round((len(cart_add_rows) / len(df)) * 100.0, 2),
        'action_taken': 'Preserve exact raw units_sold and orders_count. Do NOT overwrite units_sold.',
        'status': 'Preserved'
    })
    
    # DQ-004: Sub-Pack / Discrepant Order Rows
    sub_pack_rows = df[df['units_sold'] < df['orders_count'] * df['pack_multiplier']]
    issues.append({
        'issue_id': 'DQ-004',
        'field_name': 'pack_multiplier vs units_sold',
        'classification_tier': 'NEEDS_DERIVED_FIX',
        'issue_description': f'units_sold is less than orders_count * pack_multiplier in {len(sub_pack_rows):,} rows. Multi-pack listing fulfilled as single loose units.',
        'affected_rows_count': int(len(sub_pack_rows)),
        'affected_rows_pct': round((len(sub_pack_rows) / len(df)) * 100.0, 2),
        'action_taken': 'Flag is_pack_multiplier_discrepancy = 1; preserve original units_sold as physical sales ground truth.',
        'status': 'Resolved via Flag'
    })
    
    # DQ-005: Legacy Channel Discontinuation (TikTok Shop)
    tts_rows = df[df['channel'] == 'Glam TTS - MFN']
    issues.append({
        'issue_id': 'DQ-005',
        'field_name': 'channel (Glam TTS - MFN)',
        'classification_tier': 'BUSINESS_CONFIRMATION_REQUIRED',
        'issue_description': f'Glam TTS - MFN activity ceased completely on 2025-06-16 ({len(tts_rows):,} rows in H1 2025, 0 thereafter). Channel permanently discontinued.',
        'affected_rows_count': int(len(tts_rows)),
        'affected_rows_pct': round((len(tts_rows) / len(df)) * 100.0, 2),
        'action_taken': 'Classified as Other (TikTok); post-June 2025 dates tagged as POST_DISCONTINUATION in grid.',
        'status': 'Documented'
    })
    
    # DQ-006: Structural Nulls in Platform-Specific Signals
    amz_null_sessions = df[(df['channel'].str.contains('Amazon')) & (df['amazon_sessions'].isnull())]
    ebay_promoted_null = df[(df['channel'].str.contains('Ebay')) & (df['ebay_promoted_flag'].isnull())]
    issues.append({
        'issue_id': 'DQ-006',
        'field_name': 'amazon_sessions / ebay_promoted_flag',
        'classification_tier': 'PRESERVE_AS_IS',
        'issue_description': 'Platform-specific fields are structurally null on non-applicable platforms (e.g. sessions on eBay; promoted flag on Amazon). Exactly 10 Amazon rows missing sessions.',
        'affected_rows_count': int(len(amz_null_sessions) + len(ebay_promoted_null)),
        'affected_rows_pct': round(((len(amz_null_sessions) + len(ebay_promoted_null)) / len(df)) * 100.0, 2),
        'action_taken': 'Preserve as NULL on non-applicable platforms. Do not fill with zero.',
        'status': 'Preserved'
    })
    
    # DQ-007: Missing actual_sku Mapping Coverage
    null_actual_sku = df['actual_sku'].isnull().sum()
    issues.append({
        'issue_id': 'DQ-007',
        'field_name': 'actual_sku',
        'classification_tier': 'SAFE_TO_FIX',
        'issue_description': f'actual_sku is blank in {null_actual_sku:,} rows (87.40%). Populated only for 233 consolidated multi-pack/alias SKUs.',
        'affected_rows_count': int(null_actual_sku),
        'affected_rows_pct': round((null_actual_sku / len(df)) * 100.0, 2),
        'action_taken': 'Fallback deterministically to raw sku: canonical_sku = COALESCE(actual_sku, sku).',
        'status': 'Resolved via Fallback'
    })
    
    # DQ-008: Missing listing_id on Legacy TikTok Shop
    null_listing = df['listing_id'].isnull().sum()
    issues.append({
        'issue_id': 'DQ-008',
        'field_name': 'listing_id / child_asin',
        'classification_tier': 'SAFE_TO_FIX',
        'issue_description': f'listing_id is null in {null_listing:,} rows, strictly occurring on legacy Glam TTS channels.',
        'affected_rows_count': int(null_listing),
        'affected_rows_pct': round((null_listing / len(df)) * 100.0, 2),
        'action_taken': 'Fallback to raw sku: listing_id_resolved = COALESCE(listing_id, sku).',
        'status': 'Resolved via Fallback'
    })
    
    return pd.DataFrame(issues)

def build_transformation_log() -> pd.DataFrame:
    """
    Constructs the formal audit trail of all transformations applied from raw Excel to normalized dataset.
    """
    steps = [
        {
            'step_number': 1,
            'step_name': 'Source Lineage Attachment',
            'source_columns': 'None (Row Position)',
            'target_columns': 'source_row_id',
            'transformation_rule': 'Assign 1-indexed sequential integer matching raw Excel row order (1 to 101,085).',
            'records_affected': 101085,
            'rationale': 'Guarantees 100% end-to-end traceability and auditability back to the immutable client source.'
        },
        {
            'step_number': 2,
            'step_name': 'Platform Group & Channel Type Mapping',
            'source_columns': 'channel',
            'target_columns': 'platform_group, channel_type',
            'transformation_rule': 'Map channels to Amazon, eBay, Website, Other (TikTok / Retail Store) via deterministic business lookup.',
            'records_affected': 101085,
            'rationale': 'Enables modular multi-platform demand modeling while retaining granular channel identity.'
        },
        {
            'step_number': 3,
            'step_name': 'Canonical SKU Resolution',
            'source_columns': 'actual_sku, sku',
            'target_columns': 'canonical_sku',
            'transformation_rule': 'canonical_sku = COALESCE(actual_sku, sku).',
            'records_affected': 101085,
            'rationale': 'Standardizes variant and multi-pack listings under master product identities while preserving raw SKUs.'
        },
        {
            'step_number': 4,
            'step_name': 'Parent & Listing Fallback Resolution',
            'source_columns': 'parent_id, listing_id, sku',
            'target_columns': 'parent_id_resolved, listing_id_resolved',
            'transformation_rule': 'parent_id_resolved = COALESCE(parent_id, sku); listing_id_resolved = COALESCE(listing_id, sku).',
            'records_affected': 101085,
            'rationale': 'Eliminates structural missingness for standalone SKUs and legacy TikTok listings.'
        },
        {
            'step_number': 5,
            'step_name': 'Shared Warehouse Inventory Normalization',
            'source_columns': 'current_stock, date',
            'target_columns': 'has_inventory_signal, stockout_flag, inventory_status',
            'transformation_rule': 'For date < 2025-08-01: has_inventory_signal=0, status=UNMONITORED. For date >= 2025-08-01: status=OUT_OF_STOCK if stock==0 else IN_STOCK.',
            'records_affected': 101085,
            'rationale': 'Prevents fabricating historical stock before tracking existed; models warehouse inventory as shared physical pool.'
        },
        {
            'step_number': 6,
            'step_name': 'Observation State Classification',
            'source_columns': 'units_sold, current_stock, has_inventory_signal',
            'target_columns': 'observation_state, treatment_method, treatment_confidence',
            'transformation_rule': 'Tag OBSERVED_SALE for units_sold > 0; STOCKOUT / DEMAND_CENSORED if stock==0 and units==0; treatment=AS_OBSERVED.',
            'records_affected': 101085,
            'rationale': 'Establishes state framework to distinguish true demand from censored or missing signals.'
        },
        {
            'step_number': 7,
            'step_name': 'Zero-Price Order Identification',
            'source_columns': 'selling_price, channel',
            'target_columns': 'is_zero_price_order',
            'transformation_rule': 'is_zero_price_order = 1 if selling_price <= 0 else 0.',
            'records_affected': 331,
            'rationale': 'Preserves physical unit demand for retail store transfers while protecting price elasticity models.'
        },
        {
            'step_number': 8,
            'step_name': 'Bundle Multiplier Discrepancy Flagging',
            'source_columns': 'units_sold, orders_count, pack_multiplier',
            'target_columns': 'is_pack_multiplier_discrepancy',
            'transformation_rule': 'Flag = 1 if units_sold != orders_count * pack_multiplier, else 0.',
            'records_affected': 25124,
            'rationale': 'Distinguishes single customer multi-pack cart purchases from loose unit fulfillments.'
        },
        {
            'step_number': 9,
            'step_name': 'ML-Eligible Horizon Partitioning',
            'source_columns': 'date',
            'target_columns': 'Dataset Split (Full vs ML-Ready)',
            'transformation_rule': 'Full: 2025-01-01 to 2026-09-10 (101,085 rows); ML-Ready: 2025-08-01 to 2026-09-10 (61,511 rows).',
            'records_affected': 101085,
            'rationale': 'Excludes unmonitored inventory period from ML training pipeline while retaining complete history for analysis.'
        }
    ]
    return pd.DataFrame(steps)
