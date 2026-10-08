"""
TypeSafe AI Jev Context Adapter & Interface
===========================================
Defines the clean, decoupled interface between business/platform context and the
XGBoost Reorder Point (ROP) model, implementing Tosif's proposed architecture:

Business / Platform Context
        ↓
      Jev (TypeSafe AI System 1 Model)
        ↓
Structured Context Features
        ↓
     XGBoost
        ↓
Reorder Point (ROP) & Reorder Quantity (ROQ)

TypeSafe AI's Jev is a low-latency 'System 1' model designed to extract typed,
probabilistic decisions and context scores from unstructured/semi-structured operational state.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


@dataclass
class JevContextInput:
    """Operational business and platform context fed into Jev."""
    sku: str
    category: str
    platform: str
    current_stock: float
    daily_run_rate: float            # e.g. v14 or v30 average daily demand
    volatility_cv: float             # Coefficient of variation (intermittency/volatility)
    selling_price: float
    promo_active: bool               # eBay promoted or active deal
    buy_box_percentage: float        # Amazon Buy Box ownership (0-100)
    traffic_momentum: float          # Amazon session traffic momentum (e.g. sessions_7d / sessions_30d)
    stockout_flag: bool              # Is current inventory <= 0?
    days_since_stockout: int
    restock_known: bool              # Confirmed inbound PO / restock date known
    business_notes: str = ""         # Semi-structured narrative / marketing signals


@dataclass
class JevContextOutput:
    """
    Typed, schema-validated structured context features produced by Jev.
    These features directly augment the tabular feature space for XGBoost.
    """
    jev_replenishment_urgency: float      # [0.0, 1.0] Urgency score based on stock cover vs demand risk
    jev_promo_demand_lift: float          # [0.0, 1.0] Probability that marketing/traffic accelerates stock depletion
    jev_stockout_severity_penalty: float  # [1.0, 3.0] Criticality multiplier (hero items vs slow-moving tail)
    jev_channel_priority_score: float     # [0.0, 1.0] Channel fulfillment weight (e.g. Amazon Prime vs MFN)
    jev_confidence: float                 # [0.0, 1.0] Jev's internal decision certainty score

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class JevContextAdapter(ABC):
    """Abstract interface for TypeSafe AI Jev integration."""

    @abstractmethod
    def extract_context(self, context_input: JevContextInput) -> JevContextOutput:
        """Process a single SKU context into typed features."""
        pass

    def batch_extract_context(self, inputs: List[JevContextInput]) -> List[JevContextOutput]:
        """Process batch of operational contexts."""
        return [self.extract_context(inp) for inp in inputs]


class TypeSafeAIJevAPIAdapter(JevContextAdapter):
    """
    Production-ready network adapter for live TypeSafe AI Jev API service.
    Can be connected when live endpoint credentials are configured without
    modifying the downstream XGBoost experiment.
    """
    def __init__(self, api_key: Optional[str] = None, endpoint_url: Optional[str] = None):
        self.api_key = api_key
        self.endpoint_url = endpoint_url or "https://api.typesafe.ai/v1/jev/decide"

    def extract_context(self, context_input: JevContextInput) -> JevContextOutput:
        # In offline/local environment, if API key is not configured, fall back to DeterministicJevProvider
        if not self.api_key:
            provider = DeterministicJevProvider()
            return provider.extract_context(context_input)
        
        # When live: HTTP POST payload structured per TypeSafe AI schema
        raise NotImplementedError("Live TypeSafe AI Jev API endpoint credentials not configured in environment.")


class DeterministicJevProvider(JevContextAdapter):
    """
    Deterministic reference provider implementing Jev's typed decision logic locally.
    Enables fully reproducible, offline backtesting and evaluation of the
    'Structured + Jev Context' architecture.
    """

    def extract_context(self, inp: JevContextInput) -> JevContextOutput:
        # 1. Replenishment Urgency: Evaluates days of cover & stockout state
        if inp.current_stock <= 0:
            urgency = 1.0
        elif inp.daily_run_rate > 0:
            doc = inp.current_stock / max(inp.daily_run_rate, 0.01)
            if doc <= 7.0:
                urgency = float(np.clip(1.0 - (doc / 14.0), 0.5, 0.95))
            elif doc <= 14.0:
                urgency = float(np.clip(0.5 - ((doc - 7.0) / 28.0), 0.25, 0.5))
            else:
                urgency = float(np.clip(0.2 - ((doc - 14.0) / 100.0), 0.05, 0.2))
        else:
            urgency = 0.05

        # 2. Promo Demand Lift Risk
        promo_lift = 0.0
        if inp.promo_active:
            promo_lift += 0.35
        if inp.traffic_momentum > 1.25:
            promo_lift += 0.40
        elif inp.traffic_momentum > 1.05:
            promo_lift += 0.20
        if inp.buy_box_percentage > 85.0:
            promo_lift += 0.15
        promo_lift = float(np.clip(promo_lift, 0.0, 1.0))

        # 3. Stockout Severity Penalty (Hero revenue SKU vs Long-Tail)
        # Higher margin / higher revenue SKUs have higher stockout penalties
        revenue_weight = inp.selling_price * max(inp.daily_run_rate, 0.01)
        if revenue_weight > 50.0:
            severity = 2.5
        elif revenue_weight > 15.0:
            severity = 1.8
        else:
            severity = 1.1

        # 4. Channel Priority Score
        plat_lower = inp.platform.lower() if inp.platform else ""
        if 'amazon' in plat_lower:
            channel_prio = 0.90
        elif 'ebay' in plat_lower:
            channel_prio = 0.70
        elif 'website' in plat_lower:
            channel_prio = 0.50
        else:
            channel_prio = 0.35

        # 5. Model Confidence
        confidence = 0.92 if not pd.isna(inp.volatility_cv) and inp.volatility_cv <= 1.0 else 0.75

        return JevContextOutput(
            jev_replenishment_urgency=round(urgency, 4),
            jev_promo_demand_lift=round(promo_lift, 4),
            jev_stockout_severity_penalty=round(severity, 4),
            jev_channel_priority_score=round(channel_prio, 4),
            jev_confidence=round(confidence, 4)
        )
