"""
QA Test Harness & Result Recording Framework
============================================
Provides uniform test recording, severity categorization, and output serialization
for the Rimmel demand forecasting QA audit.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

QA_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DATA_DIR = QA_ROOT / "generated_data"
OUTPUTS_DIR = QA_ROOT / "outputs"

GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = QA_ROOT.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "rimmel_clean.db"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class QARecorder:
    """Manages recording, logging, and persisting test results."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.results: List[Dict[str, Any]] = []

    def record(
        self,
        test_id: str,
        category: str,
        test_name: str,
        status: str,  # PASS, FAIL, WARNING, NOT TESTABLE, INFORMATIONAL
        severity: str,  # CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL
        expected: str,
        actual: str,
        component: str,
        affected_sku: str = "N/A",
        root_cause: str = "N/A",
        recommended_fix: str = "N/A",
        production_affected: str = "NO",
        details: Optional[Dict[str, Any]] = None
    ):
        status_clean = status.upper().strip()
        severity_clean = severity.upper().strip()

        record_entry = {
            "test_id": test_id,
            "suite": self.suite_name,
            "category": category,
            "test_name": test_name,
            "status": status_clean,
            "severity": severity_clean,
            "expected_behavior": expected,
            "actual_behavior": actual,
            "affected_component": component,
            "affected_sku": affected_sku,
            "likely_root_cause": root_cause,
            "recommended_fix": recommended_fix,
            "production_affected": production_affected,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.results.append(record_entry)

        # Console notification
        icon = "[PASS]" if status_clean == "PASS" else ("[FAIL]" if status_clean == "FAIL" else "[WARN]")
        print(f"  {icon} {test_id}: {test_name} ({severity_clean})")
        if status_clean in ["FAIL", "WARNING"]:
            print(f"     -> Expected: {expected}")
            print(f"     -> Actual:   {actual}")

    def get_summary(self) -> Dict[str, int]:
        summary = {"PASS": 0, "FAIL": 0, "WARNING": 0, "NOT TESTABLE": 0, "INFORMATIONAL": 0}
        for r in self.results:
            s = r["status"]
            summary[s] = summary.get(s, 0) + 1
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        flat = []
        for r in self.results:
            row = {k: v for k, v in r.items() if k != "details"}
            flat.append(row)
        return pd.DataFrame(flat)


def prepare_lgbm_dataframe(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Prepares a feature dataframe for LightGBM inference:
    1. Ensures categorical columns are explicitly category dtype
    2. Converts numeric columns (especially those loaded as object/None from SQLite) to float64
    """
    out = df[feature_cols].copy()
    for col in feature_cols:
        if col in cat_cols:
            out[col] = out[col].astype("category")
        else:
            if out[col].dtype == "object":
                out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
