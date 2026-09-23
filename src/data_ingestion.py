"""
DATA INGESTION MODULE (PHASE 1)
===============================
Safely ingests the raw immutable client Excel sales data, creates an immutable
byte-level backup, assigns deterministic 1-indexed source_row_id lineage, and
extracts raw metadata without modifying any original values.
"""
import os
import shutil
import hashlib
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_SOURCE_FILE = os.path.join(BASE_DIR, 'data', 'Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx')
BACKUP_COPY_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'ORIGINAL_COPY.xlsx')

def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for immutable audit tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def ensure_immutable_backup() -> None:
    """Ensures an immutable copy of the raw source file exists in data/raw/."""
    os.makedirs(os.path.dirname(BACKUP_COPY_FILE), exist_ok=True)
    if not os.path.exists(BACKUP_COPY_FILE):
        shutil.copy2(RAW_SOURCE_FILE, BACKUP_COPY_FILE)
        print(f'[DATA INGESTION] Created immutable raw backup at: {BACKUP_COPY_FILE}')
    else:
        print(f'[DATA INGESTION] Verified existing raw backup at: {BACKUP_COPY_FILE}')

def load_raw_dataset(filepath: str = RAW_SOURCE_FILE) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads the immutable raw Excel file into memory, attaches source_row_id lineage,
    and returns the dataframe along with raw audit metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Raw dataset not found at: {filepath}')
        
    ensure_immutable_backup()
    
    file_size_bytes = os.path.getsize(filepath)
    file_hash = compute_file_hash(filepath)
    
    print(f'[DATA INGESTION] Ingesting raw Excel: {filepath} ({file_size_bytes:,} bytes, SHA-256: {file_hash[:12]}...)')
    df_raw = pd.read_excel(filepath)
    
    # Preserve 1-indexed source row number for 100% lineage traceability
    df_raw.insert(0, 'source_row_id', range(1, len(df_raw) + 1))
    
    raw_metadata = {
        'source_file': filepath,
        'backup_copy': BACKUP_COPY_FILE,
        'sha256_hash': file_hash,
        'file_size_bytes': file_size_bytes,
        'ingestion_timestamp': datetime.now().isoformat(),
        'total_rows_imported': len(df_raw),
        'total_cols_imported': len(df_raw.columns),
        'columns_list': list(df_raw.columns)
    }
    
    print(f'[DATA INGESTION] Successfully loaded {len(df_raw):,} rows and {len(df_raw.columns)} columns.')
    return df_raw, raw_metadata

if __name__ == '__main__':
    df, meta = load_raw_dataset()
    print('Sample raw row with lineage:')
    print(df.iloc[0][['source_row_id', 'date', 'sku', 'channel', 'units_sold', 'selling_price']])
