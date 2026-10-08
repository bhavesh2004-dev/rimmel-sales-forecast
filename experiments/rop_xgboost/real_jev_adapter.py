"""
Real TypeSafe Jev OpenRouter Adapter with Persistent Disk Caching
=================================================================
Connects to OpenRouter's Decisions API (POST https://openrouter.ai/api/alpha/decisions)
for model '~typesafe/jev-latest' with automatic disk caching and token tracking.

Security & Safety:
- Loads API key strictly from local .env without printing or leaking.
- Never serializes credentials to cache or logs.
- Automatic rate-limit handling and graceful fallbacks.
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dotenv import load_dotenv

import threading

from experiments.rop_xgboost.jev_cache import JevCache

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_NAME = "~typesafe/jev-latest"
ENV_PATH = Path(__file__).resolve().parent / ".env"


class RealJevOpenRouterAdapter:
    """Client adapter for real TypeSafe Jev via OpenRouter with caching."""

    def __init__(self, cache: Optional[JevCache] = None, timeout: float = 30.0):
        self.cache = cache or JevCache()
        self.timeout = timeout
        self.api_calls_made = 0
        self.cache_hits = 0
        self.retries_count = 0
        self.failures_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.lock = threading.Lock()

        load_dotenv(dotenv_path=ENV_PATH, override=True)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(f"OPENROUTER_API_KEY missing in {ENV_PATH}. Cannot execute real Jev requests.")

    def get_decision(self, date: str, sku: str, platform: str, state_text: str) -> Dict[str, Any]:
        """
        Retrieves real TypeSafe Jev decisions for the given context.
        Checks persistent cache first; only dispatches an API request if unvisited.
        """
        # 1. Check Cache
        cached = self.cache.get(date, sku, platform, state_text)
        if cached is not None:
            with self.lock:
                self.cache_hits += 1
            return self._parse_response(cached, is_cached=True)

        # 2. Construct Payload
        payload = {
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bhavesh2004-dev/rimmel-sales-forecast",
            "X-Title": "Rimmel ROP Replenishment Experiment - Phase 2"
        }

        # 3. Dispatch Single API Call with Exponential Backoff Retries
        max_retries = 4
        backoff_delay = 1.5
        resp = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(OPENROUTER_DECISIONS_URL, headers=headers, json=payload, timeout=self.timeout)
                with self.lock:
                    self.api_calls_made += 1

                if resp.status_code == 200:
                    break
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (backoff_delay * (2 ** attempt))
                    with self.lock:
                        self.retries_count += 1
                    time.sleep(sleep_time)
                elif resp.status_code in [500, 502, 503, 504]:
                    with self.lock:
                        self.retries_count += 1
                    time.sleep(backoff_delay * (2 ** attempt))
                else:
                    with self.lock:
                        self.failures_count += 1
                    raise RuntimeError(f"OpenRouter API error HTTP {resp.status_code}: {resp.text[:200]}")
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt < max_retries:
                    with self.lock:
                        self.retries_count += 1
                    time.sleep(backoff_delay * (2 ** attempt))
                else:
                    with self.lock:
                        self.failures_count += 1
                    raise RuntimeError(f"OpenRouter API request failed after {max_retries} retries: {e}")

        if resp is None or resp.status_code != 200:
            with self.lock:
                self.failures_count += 1
            raise RuntimeError(f"OpenRouter API error HTTP {resp.status_code if resp else 'None'}: {resp.text[:200] if resp else 'No response'}")

        resp_data = resp.json()

        # Update usage metrics
        usage = resp_data.get("usage", {})
        cost = usage.get("cost", 0.0)
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)

        with self.lock:
            self.total_cost_usd += cost
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok

        # 4. Save to Disk Cache
        self.cache.set(date, sku, platform, state_text, resp_data)

        return self._parse_response(resp_data, is_cached=False)

    def _parse_response(self, resp_data: Dict[str, Any], is_cached: bool) -> Dict[str, Any]:
        """Parses raw Jev decisions into structured numeric features."""
        answers = resp_data.get("answers", {})

        # Noul: Urgency Probability
        noul_info = answers.get("urgent_replenishment", {})
        urgency_prob = float(noul_info.get("noul", 0.5))

        # Score: Replenishment Risk
        score_info = answers.get("replenishment_risk", {})
        risk_score = float(score_info.get("score", 1.0))
        risk_conf = float(score_info.get("confidence", 0.8))

        # Choice: Replenishment Priority
        choice_info = answers.get("replenishment_priority", {})
        priority_label = str(choice_info.get("choice", "MEDIUM")).upper()
        prio_conf = float(choice_info.get("confidence", 0.8))

        # One-hot priority flags for XGBoost
        prio_high = 1.0 if priority_label == "HIGH" else 0.0
        prio_med = 1.0 if priority_label == "MEDIUM" else 0.0
        prio_low = 1.0 if priority_label == "LOW" else 0.0

        # Numeric priority scale (for ablation testing)
        prio_code = 2.0 if priority_label == "HIGH" else (1.0 if priority_label == "MEDIUM" else 0.0)

        # Composite model confidence
        overall_conf = (risk_conf + prio_conf) / 2.0

        return {
            "jev_urgent_replenishment_prob": urgency_prob,
            "jev_replenishment_risk_score": risk_score,
            "jev_replenishment_priority_label": priority_label,
            "jev_prio_HIGH": prio_high,
            "jev_prio_MEDIUM": prio_med,
            "jev_prio_LOW": prio_low,
            "jev_replenishment_priority_code": prio_code,
            "jev_confidence": overall_conf,
            "is_cached": is_cached,
            "model_version": resp_data.get("model", "unknown")
        }
