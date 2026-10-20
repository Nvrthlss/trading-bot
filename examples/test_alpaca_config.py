"""
Quick test: can we read Alpaca credentials from .env?
"""

from src.config import ALPACA_API_KEY_ID, ensure_alpaca_credentials

ensure_alpaca_credentials()

print("Credentials loaded successfully.")
print(f"Key ID starts with: {ALPACA_API_KEY_ID[:5]}...")
print(f"Key ID length: {len(ALPACA_API_KEY_ID)} chars")