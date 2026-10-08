"""
Reorder Point (ROP) & Reorder Quantity (ROQ) Formulation
=========================================================
Formal mathematical definitions, target variable specifications, and explicit audit
of missing business requirements for the ROP XGBoost experiment.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# 1. EXPLICIT AUDIT OF MISSING BUSINESS REQUIREMENTS
# =============================================================================
MISSING_BUSINESS_REQUIREMENTS = {
    "Supplier Lead Time (L)": (
        "MISSING in client files. The dataset records restock_date for only 763 rows, "
        "but has zero records of purchase order creation dates, supplier dispatch times, "
        "or manufacturer transit delays. (Handled via parameterized scenario: L=7 and L=14 days)."
    ),
    "Supplier Minimum Order Quantity (MOQ)": (
        "MISSING in client files. While pack_multiplier exists for customer packs, supplier "
        "case-pack minimums and vendor batch constraints are absent."
    ),
    "Carrying / Holding Cost & Ordering Cost": (
        "MISSING in client files. Economic Order Quantity (EOQ) requires annual holding cost % "
        "and fixed PO placement cost, which are not provided."
    ),
    "Inbound Purchase Orders in Transit (Pipeline Inventory)": (
        "MISSING in client files. The data reflects only current on-hand warehouse stock, "
        "with no tracking of confirmed open POs currently on water/in transit."
    ),
    "Executive Service Level SLA Target": (
        "UNDEFINED by client. Classical inventory planning requires an executive SLA "
        "(e.g., 90%, 95%, or 99% Cycle Service Level). Standard default of 95% is benchmarked."
    )
}


def audit_missing_requirements() -> Dict[str, str]:
    """Returns the explicit registry of missing supply-chain data requirements."""
    return MISSING_BUSINESS_REQUIREMENTS


# =============================================================================
# 2. MATHEMATICAL ROP & REPLENISHMENT FORMULATION
# =============================================================================
"""
CLASSICAL FORMULATION:
  ROP = Expected Lead Time Demand (LTD) + Safety Stock (SS)
  LTD = L * d_bar
  SS  = z_alpha * sigma_LTD

LIMITATIONS OF CLASSICAL FORMULATION ON RIMMEL DATA:
  - Assumes demand is independent and normally distributed (i.i.d.).
  - Over 75% of Rimmel daily SKU-platform series have zero sales (intermittent/sparse).
  - Volatility is non-Gaussian with sudden spikes and promotions.

MACHINE LEARNING QUANTILE ROP FORMULATION (XGBoost):
  Instead of assuming Gaussian distribution and fitting mean + static variance,
  XGBoost directly estimates the conditional (1 - alpha) quantile of cumulative
  lead-time demand:

    Target_LTD(t, L) = sum_{k=1}^{L} observed_units_sold(t + k)

  Model learns:
    ROP_hat(t, L) = Q_0.95( Target_LTD(t, L) | X_t )

  This guarantees that:
    P( Actual Demand over L <= ROP_hat ) >= 95%
  directly matching the 95% Cycle Service Level without parametric distributional assumptions.
"""


def compute_reorder_trigger(inventory_position: float, rop: float) -> bool:
    """
    Evaluates whether a purchase order must be placed at decision date t.
    Trigger condition: Inventory Position <= Reorder Point.
    """
    return bool(inventory_position <= max(rop, 1.0))


def compute_reorder_quantity(
    inventory_position: float,
    rop: float,
    daily_run_rate: float,
    target_cover_days: int = 30,
    moq: int = 1
) -> int:
    """
    Computes Reorder Quantity (ROQ) under an Order-Up-To Level (s, S) policy.
    S = Maximum Target Stock = ROP + (Target Cover Days * Expected Daily Run Rate)
    ROQ = max(0, S - Inventory Position) rounded up to MOQ.
    """
    if not compute_reorder_trigger(inventory_position, rop):
        return 0

    order_up_to_level = rop + (target_cover_days * max(daily_run_rate, 0.05))
    raw_qty = max(0.0, order_up_to_level - inventory_position)
    
    # Enforce minimum order quantity if applicable
    if raw_qty > 0 and raw_qty < moq:
        raw_qty = float(moq)
        
    return int(np.ceil(raw_qty))
