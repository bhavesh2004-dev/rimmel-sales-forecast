"""
Jev Response Caching Layer
==========================
Persistent, content-addressable disk cache for real TypeSafe Jev API responses.
Prevents duplicate API calls, eliminates unnecessary API credit spend,
and enables deterministic, leakage-free reproducibility.

Cache Key:
  sha256(f"{decision_date}|{canonical_sku}|{platform_group}|{context_text}")

Security Guarantee:
  Never stores or serializes the OpenRouter API key.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "jev_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_cache_key(date: str, sku: str, platform: str, context_text: str) -> str:
    """Computes a deterministic SHA-256 hash for the given decision context."""
    raw = f"{date}|{sku}|{platform}|{context_text.strip()}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class JevCache:
    """Manages reading and writing typed Jev decisions to disk cache."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, date: str, sku: str, platform: str, context_text: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached response if present; otherwise returns None."""
        key = compute_cache_key(date, sku, platform, context_text)
        fp = self._file_path(key)
        if fp.exists():
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except Exception:
                return None
        return None

    def set(self, date: str, sku: str, platform: str, context_text: str, response_data: Dict[str, Any]) -> str:
        """Saves a response to disk cache. Returns the cache key."""
        key = compute_cache_key(date, sku, platform, context_text)
        fp = self._file_path(key)

        # Sanitization: Ensure NO authorization tokens exist in the stored dictionary
        clean_data = {k: v for k, v in response_data.items() if 'key' not in k.lower() and 'auth' not in k.lower()}
        clean_data['cache_key'] = key
        clean_data['date'] = date
        clean_data['sku'] = sku
        clean_data['platform'] = platform

        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2)

        return key

    def count(self) -> int:
        """Returns the number of cached decisions currently stored."""
        return len(list(self.cache_dir.glob("*.json")))
