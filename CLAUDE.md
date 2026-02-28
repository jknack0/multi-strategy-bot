# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-strategy intraday trading system for MES (Micro E-mini S&P 500) futures. Uses `uv` for dependency management, QuestDB for time-series storage, and Interactive Brokers for execution. Python 3.11+.

## Commands

```bash
# Install dependencies (creates .venv automatically)
uv sync --all-extras

# Run all tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_indicators.py -v

# Run a single test function
uv run pytest tests/test_indicators.py::test_bollinger_basic -v

# Lint
uv run ruff check .

# Start QuestDB (required for data storage/queries)
docker-compose up -d
```

## Architecture

### Data Pipeline
- **QuestDB** (`data/questdb_client.py`) stores bars via ILP ingestion (port 9009) and queries via PostgreSQL wire protocol (port 8812). Table: `ohlcv`, partitioned by month with dedup on (timestamp, symbol, bar_size)
- **BarAggregator** (`data/aggregator.py`) converts real-time 5-second bars into 1-min and 5-min bars with callback-based emission. Handles futures session boundaries (18:00 ET) and VWAP reset (9:30 ET)

### Strategy Pattern
All strategies follow the same event-driven interface:
- `on_bar(timestamp, open_, high, low, close, volume)` processes a single bar, returns a `TradeRecord` if a trade closed
- `generate_signals(df, vix_series)` runs the full bar-by-bar loop over a DataFrame for backtesting
- `reset()` clears all state for a new backtest run
- Parameters stored as `DEFAULT_PARAMS` dict, overridable via constructor or JSON files in `config/`

**Strategy A** (`strategies/mean_reversion.py`): Bollinger Band + VWAP mean reversion on 5-min bars. Entries when price touches BB band and deviates from VWAP. Has three signal generation paths: `generate_signals` (iterrows), `generate_signals_fast` (numpy arrays), and `generate_signals_vectorized` (pure vectorized, no position management). The `precompute_indicators()` function pre-computes shared indicators across parameter grid searches.

**Strategy B** (`strategies/orb_strategy.py`): VIX-adaptive Opening Range Breakout. Adapts OR duration to VIX regime (5/15/30 min). Filters on relative volume, VWAP slope, and OR width vs ATR. One trade per day max.

### Risk Management (portfolio-level)
- **RiskManager** (`risk/risk_manager.py`): Capital allocation across strategies (default 40/30/30 split), quarter-Kelly position sizing, exposure tracking
- **CircuitBreaker** (`risk/circuit_breaker.py`): Three-level daily loss limits (1%/2%/3% of capital) escalating from REDUCED to NO_NEW_ENTRIES to HALTED. Weekly 5% limit reduces sizes for following week
- **DynamicStops** (`risk/dynamic_stops.py`): ATR-based trailing stops with breakeven logic

### Backtesting
- **BacktestEngine** (`backtesting/backtest_engine.py`): VectorBT-based. Load data, set entry/exit signals, run. Applies configurable slippage (tick-based) and commissions
- **CPCVValidator** (`backtesting/cpcv_validator.py`): Combinatorial Purged Cross-Validation (Lopez de Prado). Computes Probability of Backtest Overfitting (PBO). PBO > 0.5 suggests overfitting
- **Monte Carlo** (`backtesting/monte_carlo.py`): Path simulation for drawdown confidence intervals
- **Metrics** (`backtesting/metrics.py`): Pure functions for Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate, expectancy

### Indicators
All indicators are stateful, incremental (call `.update()` per bar), and return `None` until enough data accumulates. Located in `indicators/`.

### Configuration
- `config/constants.py`: Instrument specs (MES/ES/MNQ tick sizes, margins, costs), market session times (US/Eastern), VIX regime thresholds and parameter adjustments
- `config/settings.py`: Environment-loaded settings (API keys, connection strings, backtest defaults) via `.env`
- `config/strategy_*_params.json`: Serialized strategy parameter overrides

## Key Conventions

- All timestamps are handled in US/Eastern timezone. VWAP resets at 9:30 ET (RTH open). Futures session boundary at 18:00 ET
- All positions flatten by 15:55 ET (time stop)
- Trading window: 10:00-14:00 ET (no entries outside this window)
- Backtest cost model: commission $0.62/side + 0.25 ticks slippage per side for MES
- VIX regime gates: EXTREME (>35) blocks Strategy A entries entirely, HIGH (25-35) widens BB bands and cuts position size 50%
- Research notebooks in `notebooks/` are `.py` files (not `.ipynb`), likely used with VS Code's interactive Python/Jupytext
