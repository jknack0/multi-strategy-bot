"""
Centralized configuration loaded from environment variables.

All API keys, connection strings, and tunable parameters are defined here.
Uses python-dotenv to load from .env file if present.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# ── Interactive Brokers ──────────────────────────────────────────────────────
IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT: int = int(os.getenv("IB_PORT", "4002"))  # 4002=paper, 4001=live
IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))
IB_TIMEOUT: int = int(os.getenv("IB_TIMEOUT", "30"))
IB_READONLY: bool = os.getenv("IB_READONLY", "False").lower() == "true"

# ── QuestDB ──────────────────────────────────────────────────────────────────
QUESTDB_HOST: str = os.getenv("QUESTDB_HOST", "127.0.0.1")
QUESTDB_ILP_PORT: int = int(os.getenv("QUESTDB_ILP_PORT", "9009"))
QUESTDB_PG_PORT: int = int(os.getenv("QUESTDB_PG_PORT", "8812"))
QUESTDB_HTTP_PORT: int = int(os.getenv("QUESTDB_HTTP_PORT", "9000"))
QUESTDB_PG_USER: str = os.getenv("QUESTDB_PG_USER", "admin")
QUESTDB_PG_PASSWORD: str = os.getenv("QUESTDB_PG_PASSWORD", "quest")
QUESTDB_PG_DATABASE: str = os.getenv("QUESTDB_PG_DATABASE", "qdb")

# ── Telegram Alerts ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Trading Parameters ───────────────────────────────────────────────────────
DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "MES")
BAR_SIZE_RT: str = os.getenv("BAR_SIZE_RT", "5 secs")
COMMISSION_PER_SIDE: float = float(os.getenv("COMMISSION_PER_SIDE", "0.62"))
SLIPPAGE_TICKS: float = float(os.getenv("SLIPPAGE_TICKS", "0.25"))

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Backtest Defaults ────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL: float = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000.0"))
BACKTEST_RISK_FREE_RATE: float = float(os.getenv("BACKTEST_RISK_FREE_RATE", "0.05"))
CPCV_N_GROUPS: int = int(os.getenv("CPCV_N_GROUPS", "6"))
CPCV_K_TEST: int = int(os.getenv("CPCV_K_TEST", "2"))
MONTE_CARLO_PATHS: int = int(os.getenv("MONTE_CARLO_PATHS", "1000"))
