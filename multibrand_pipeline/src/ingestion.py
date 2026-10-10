"""
Multi-Brand Ingestion Module
Dynamically ingests Excel workbooks or CSV files, resolves column aliases,
validates required business schemas, and unifies raw transaction records.
Principle: Brand as Data.
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class IngestionError(Exception):
    """Raised when ingestion validation fails."""
    pass

def normalize_header(header: str) -> str:
    """Normalize column header for alias matching."""
    return re.sub(r'[^a-z0-9]', '', str(header).lower().strip())

def resolve_column_mapping(columns: List[str], schema_aliases: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Maps incoming raw column names to canonical internal concepts.
    Returns dict: {raw_col_name: canonical_concept}
    """
    mapping = {}
    norm_incoming = {col: normalize_header(col) for col in columns}
    
    for concept, aliases in schema_aliases.items():
        norm_aliases = [normalize_header(a) for a in aliases]
        for raw_col, norm_col in norm_incoming.items():
            if raw_col not in mapping and norm_col in norm_aliases:
                mapping[raw_col] = concept
                break
                
    return mapping

def ingest_excel(
    file_path: str,
    schema_aliases: Dict[str, List[str]],
    default_brand: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests an Excel workbook dynamically inspecting all sheets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    logger.info(f"Ingesting Excel file: {file_path}")
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    logger.info(f"Discovered {len(sheet_names)} sheet(s): {sheet_names}")
    
    all_dfs = []
    metadata = {
        'source_file': file_path,
        'sheets_processed': [],
        'total_raw_rows': 0,
        'brands_detected': set(),
        'validation_warnings': []
    }
    
    for sheet in sheet_names:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
        if df_sheet.empty:
            logger.warning(f"Sheet '{sheet}' is empty. Skipping.")
            continue
            
        col_map = resolve_column_mapping(list(df_sheet.columns), schema_aliases)
        inv_map = {concept: raw_col for raw_col, concept in col_map.items()}
        
        # Check required fields: date, sku, quantity, channel
        required = ['date', 'sku', 'quantity', 'channel']
        missing_req = [r for r in required if r not in inv_map]
        
        if missing_req:
            msg = f"Sheet '{sheet}' missing required concepts: {missing_req}. Columns found: {list(df_sheet.columns)}"
            # If sheet doesn't look like a transaction sheet, log warning and skip
            logger.warning(msg)
            metadata['validation_warnings'].append(msg)
            continue
            
        # Rename identified columns
        df_clean = df_sheet.rename(columns=col_map)
        
        # Brand detection
        if 'brand' in df_clean.columns and df_clean['brand'].notnull().any():
            # Brand is present as a column
            df_clean['raw_brand'] = df_clean['brand'].astype(str)
        else:
            # Infer brand from sheet name or file name or default
            inferred = default_brand
            if not inferred:
                for b_name in ['Rimmel', 'Max Factor', 'MaxFactor']:
                    if b_name.lower() in sheet.lower() or b_name.lower() in file_path.lower():
                        inferred = b_name
                        break
            if not inferred:
                inferred = sheet.strip()
            df_clean['raw_brand'] = inferred
            
        df_clean['raw_source_sheet'] = sheet
        df_clean['raw_source_file'] = os.path.basename(file_path)
        
        # Track brands
        detected = df_clean['raw_brand'].unique().tolist()
        metadata['brands_detected'].update(detected)
        metadata['sheets_processed'].append(sheet)
        metadata['total_raw_rows'] += len(df_clean)
        
        all_dfs.append(df_clean)
        
    if not all_dfs:
        raise IngestionError(f"No valid transaction sheets found in {file_path}")
        
    combined = pd.concat(all_dfs, ignore_index=True)
    metadata['brands_detected'] = list(metadata['brands_detected'])
    logger.info(f"Successfully ingested {len(combined):,} rows across brands: {metadata['brands_detected']}")
    
    return combined, metadata

def ingest_csv_or_excel(
    file_path: str,
    schema_aliases: Dict[str, List[str]],
    default_brand: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Unified entrypoint supporting Excel, CSV, and Parquet input files."""
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
        col_map = resolve_column_mapping(list(df.columns), schema_aliases)
        df_clean = df.rename(columns=col_map)
        if 'brand' in df_clean.columns and df_clean['brand'].notnull().any():
            df_clean['raw_brand'] = df_clean['brand'].astype(str)
        elif 'brand_id' in df_clean.columns and df_clean['brand_id'].notnull().any():
            df_clean['raw_brand'] = df_clean['brand_id'].astype(str)
        else:
            df_clean['raw_brand'] = default_brand or 'UNKNOWN_BRAND'
        df_clean['raw_source_sheet'] = 'PARQUET_ROOT'
        df_clean['raw_source_file'] = os.path.basename(file_path)
        meta = {
            'source_file': file_path,
            'sheets_processed': ['PARQUET_ROOT'],
            'total_raw_rows': len(df_clean),
            'brands_detected': list(df_clean['raw_brand'].unique()),
            'validation_warnings': []
        }
        return df_clean, meta
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        col_map = resolve_column_mapping(list(df.columns), schema_aliases)
        df_clean = df.rename(columns=col_map)
        if 'brand' in df_clean.columns and df_clean['brand'].notnull().any():
            df_clean['raw_brand'] = df_clean['brand'].astype(str)
        else:
            df_clean['raw_brand'] = default_brand or 'UNKNOWN_BRAND'
        df_clean['raw_source_sheet'] = 'CSV_ROOT'
        df_clean['raw_source_file'] = os.path.basename(file_path)
        meta = {
            'source_file': file_path,
            'sheets_processed': ['CSV_ROOT'],
            'total_raw_rows': len(df_clean),
            'brands_detected': list(df_clean['raw_brand'].unique()),
            'validation_warnings': []
        }
        return df_clean, meta
    else:
        return ingest_excel(file_path, schema_aliases, default_brand)
