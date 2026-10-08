"""
TypeSafe Jev Latest Connectivity Test via OpenRouter Decisions API
==================================================================
Isolated connectivity test for TypeSafe AI's Jev model using OpenRouter's
dedicated Decisions API (/api/alpha/decisions).

Safety & Governance Constraints:
- Isolated to experimental evaluation only.
- Exp6 production forecasting system is NOT modified.
- DeterministicJevProvider is NOT deleted or altered.
- Exactly ONE test request is dispatched with synthetic context.
- API keys are NEVER printed or logged.
"""
import os
import sys
import json
from pathlib import Path
import requests
from dotenv import load_dotenv

# Endpoint and model constants
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_NAME = "~typesafe/jev-latest"

# Synthetic test state and query
TEST_STATE = "A product currently has 5 units in stock and recent demand is approximately 8 units per day."
QUESTION_KEY = "urgent_replenishment"
QUESTION_INSTRUCTIONS = "Is urgent replenishment likely to be required?"


def load_api_key() -> tuple[str, Path]:
    """
    Safely load OPENROUTER_API_KEY from .env without printing or exposing secrets.
    Looks in current directory, script directory, and project root.
    """
    script_dir = Path(__file__).resolve().parent
    env_paths = [
        script_dir / ".env",
        Path.cwd() / ".env",
        script_dir.parent.parent / ".env"
    ]
    
    target_env = None
    for p in env_paths:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            target_env = p
            break
            
    if target_env is None:
        target_env = script_dir / ".env"
        load_dotenv(dotenv_path=target_env, override=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return api_key, target_env


def run_connectivity_test():
    api_key, env_path = load_api_key()

    print("==================================================")
    print("OPENROUTER / TYPESAFE JEV CONNECTIVITY REPORT")
    print("==================================================")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: /api/alpha/decisions")

    # Safe validation check
    if not api_key:
        print("OpenRouter connection: FAIL")
        print("HTTP status: N/A (Missing OPENROUTER_API_KEY)")
        print("Jev response received: NO")
        print("Structured decision received: NO")
        print("--------------------------------------------------")
        print("[!] Safe Reason: OPENROUTER_API_KEY is empty or not set.")
        print(f"    Target .env location: {env_path}")
        print("    Please paste your OpenRouter API key into the .env file:")
        print("      OPENROUTER_API_KEY=your_key_here")
        print("    Then re-run this script to execute the single connectivity test.")
        print("==================================================")
        return False

    # Headers - key is never logged or printed
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bhavesh2004-dev/rimmel-sales-forecast",
        "X-Title": "Rimmel ROP Replenishment Experiment",
    }

    # Structured Decisions API payload
    payload = {
        "model": MODEL_NAME,
        "state": TEST_STATE,
        "questions": {
            QUESTION_KEY: {
                "type": "noul",
                "instructions": QUESTION_INSTRUCTIONS
            }
        }
    }

    print(f"Synthetic State: \"{TEST_STATE}\"")
    print(f"Noul Question: \"{QUESTION_INSTRUCTIONS}\"")
    print("Sending single API request to OpenRouter Decisions endpoint...")

    try:
        response = requests.post(
            OPENROUTER_DECISIONS_URL,
            headers=headers,
            json=payload,
            timeout=30.0
        )
        status_code = response.status_code
        print(f"HTTP status: {status_code}")

        if status_code == 200:
            data = response.json()
            # Inspect structure
            answers = data.get("answers", {})
            has_jev_response = bool(data)
            has_structured_decision = QUESTION_KEY in answers

            print(f"OpenRouter connection: PASS")
            print(f"Jev response received: {'YES' if has_jev_response else 'NO'}")
            print(f"Structured decision received: {'YES' if has_structured_decision else 'NO'}")

            if has_structured_decision:
                decision_info = answers[QUESTION_KEY]
                prob = decision_info.get("noul", decision_info.get("probability", "N/A"))
                print("--------------------------------------------------")
                print(f"Structured Decision Output ({QUESTION_KEY}):")
                print(f"  Primitive Type: {decision_info.get('type', 'noul')}")
                print(f"  Replenishment Urgency Probability: {prob}")
                if "usage" in data:
                    print(f"  Usage: {data['usage']}")
            else:
                print("--------------------------------------------------")
                print(f"Response Content: {json.dumps(data, indent=2)}")

            print("==================================================")
            return True

        else:
            # Handle non-200 responses safely without printing sensitive headers
            print("OpenRouter connection: FAIL")
            print("Jev response received: NO")
            print("Structured decision received: NO")
            print("--------------------------------------------------")
            try:
                err_json = response.json()
                safe_err = err_json.get("error", {}).get("message", response.text[:200])
            except Exception:
                safe_err = response.text[:200]
            print(f"[!] Safe Error Message: {safe_err}")
            
            if status_code == 401:
                print("    Likely Cause: Invalid or expired OpenRouter API key.")
            elif status_code == 404:
                print("    Likely Cause: The Decisions API endpoint or model alias is unavailable or in private alpha.")
            elif status_code == 429:
                print("    Likely Cause: Rate limit or credit limit reached.")
            else:
                print(f"    Likely Cause: API returned status {status_code}.")

            print("==================================================")
            return False

    except requests.exceptions.Timeout:
        print("OpenRouter connection: FAIL")
        print("HTTP status: Timeout (>30s)")
        print("Jev response received: NO")
        print("Structured decision received: NO")
        print("Safe Error: Request timed out while connecting to openrouter.ai.")
        print("==================================================")
        return False
    except requests.exceptions.RequestException as e:
        print("OpenRouter connection: FAIL")
        print("HTTP status: Connection Error")
        print("Jev response received: NO")
        print("Structured decision received: NO")
        print(f"Safe Error: {type(e).__name__}")
        print("==================================================")
        return False


if __name__ == "__main__":
    success = run_connectivity_test()
    sys.exit(0 if success else 1)
