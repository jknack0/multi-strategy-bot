# Intraday Trading Bot

Multi-strategy intraday trading system for MES (Micro E-mini S&P 500) futures.

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all dependencies (creates .venv automatically)
uv sync --all-extras

# 3. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start QuestDB
docker-compose up -d

# 5. Run tests
uv run pytest -v

# 6. Fetch historical data (requires Polygon API key)
uv run python -c "from data.polygon_fetcher import PolygonFetcher; PolygonFetcher().fetch_and_store()"
```

## Project Structure

```
config/          Configuration and instrument constants
data/            Data connectors (IB, Polygon, QuestDB, aggregator)
indicators/      Technical indicators (VWAP, Bollinger, ATR, VIX regime)
backtesting/     Backtesting engine, CPCV validation, Monte Carlo simulation
strategies/      Trading strategies (Sprint 1-3)
risk/            Risk management (Sprint 2)
execution/       Order execution (Sprint 4)
notebooks/       Research notebooks
tests/           Test suite
```

## Testing

```bash
uv run pytest -v
uv run pytest tests/test_indicators.py -v    # Indicator tests only
uv run pytest tests/test_cpcv.py -v          # CPCV validation tests
```
