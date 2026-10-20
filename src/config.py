"""
Project configuration — loads credentials from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# Alpaca credentials
ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")


def ensure_alpaca_credentials():
    if not ALPACA_API_KEY_ID or not ALPACA_API_SECRET:
        raise RuntimeError(
            "Alpaca credentials not found. Create a .env file in the project root with:\n"
            "  ALPACA_API_KEY_ID=your_key_id\n"
            "  ALPACA_API_SECRET=your_secret_key"
        )