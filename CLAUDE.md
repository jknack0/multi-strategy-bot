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

# Sync local parquet data to Supabase (bulk historical backfill)
uv run python scripts/sync_market_data.py data/MES_1min.parquet --symbol MES --timeframe 1m

# EOD sync: push today's QuestDB bars to Supabase (run after market close)
uv run python scripts/eod_sync.py
uv run python scripts/eod_sync.py --date 2026-02-28 --days 5
```

## Architecture

### Data Pipeline
- **QuestDB** (`data/questdb_client.py`) stores bars via ILP ingestion (port 9009) and queries via PostgreSQL wire protocol (port 8812). Table: `ohlcv`, partitioned by month with dedup on (timestamp, symbol, bar_size)
- **BarAggregator** (`data/aggregator.py`) converts real-time 5-second bars into 1-min and 5-min bars with callback-based emission. Handles futures session boundaries (18:00 ET) and VWAP reset (9:30 ET)
- **Supabase** (`data/supabase_client.py`) analytics warehouse for historical market data, trade history, and performance snapshots. Write-mostly layer for dashboards and model training. Tables: `market_data`, `strategies`, `trades`, `round_trips`, `strategy_snapshots`

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

### Supabase Schema
Tables in the analytics warehouse (Supabase). The Python client uses `service_role` key to bypass RLS.

- **`market_data`**: OHLCV bars partitioned by timeframe (`1s`, `1m`, `5m`, `15m`, `1h`, `1d`). PK: `(symbol, timeframe, open_time)`. Columns: `open`, `high`, `low`, `close` (double precision), `volume`, `vwap`, `trade_count`, `source`
- **`market_data_l1`**: Level 1 tick data. PK: `(symbol, ts)`. Columns: `bid`, `ask`, `bid_size`, `ask_size`, `last_price`, `last_size`, `source`
- **`dollar_bars`**: Variable-length dollar bars. Unique on `(symbol, bar_index)`. Includes `dollar_value`, `threshold`, `buy_volume`, `sell_volume`
- **`strategies`**: Strategy registry. PK: `id` (text, e.g. `bb_vwap_mr_v1`, `orb_vix_v1`). Columns: `name`, `version`, `config` (jsonb), `status` (`active`/`paper`/`retired`)
- **`model_artifacts`**: Trained model params (HMM, BOCPD, PCA). FK to `strategies(id)`. Columns: `model_type`, `artifact` (jsonb), `metrics` (jsonb), `is_active`
- **`regime_predictions`**: HMM/BOCPD regime output. Unique on `(symbol, bar_index)`. Columns: `regime_name`, `prob_mean_rev`, `prob_trending`, `prob_volatile`, `confidence`, `bocpd_cp_prob`
- **`signals`**: Strategy signals. FK to `strategies(id)`. Columns: `direction`, `strength`, `entry_price`, `stop_loss`, `profit_target`, `regime`, `features` (jsonb)
- **`trades`**: Individual executions. FK to `strategies(id)` and optional FK to `signals(id)`. Columns: `side` (`buy`/`sell`), `quantity`, `price`, `commission` (numeric), `executed_at`, `broker`
- **`round_trips`**: Entry+exit pairs. FK to `strategies(id)`, `trades(id)` for entry/exit. Columns: `direction`, `entry_price`, `exit_price`, `gross_pnl`, `net_pnl`, `hold_duration` (interval), `exit_reason`, `max_adverse`, `max_favorable`
- **`strategy_snapshots`**: Daily performance snapshots. Unique on `(strategy_id, snapshot_date, symbol)`. Columns: `sharpe`, `sortino`, `max_drawdown`, `profit_factor`, `win_rate`, `total_trades`, `avg_pnl`
- **`walk_forward_results`**: Walk-forward validation runs. FK to `strategies(id)`. Columns: `n_windows`, `sharpe_mean`, `is_passing`, `window_details` (jsonb)

### Live Execution
- **LiveRunner** (`execution/live_runner.py`): Paper trading orchestrator. Streams 1-min bars from Databento, feeds both strategies, tracks PnL internally, persists to QuestDB/Supabase
- **DatabentoLiveConnector** (`data/databento_live.py`): Subscribes to `GLBX.MDP3` / `MES.FUT` / `ohlcv-1m`, dispatches `(timestamp, o, h, l, c, v)` callbacks
- Strategy IDs for Supabase: `bb_vwap_mr_v1` (Strategy A), `orb_vix_v1` (Strategy B)
- Graceful degradation: QuestDB and Supabase failures don't crash the runner

## Key Conventions

- All timestamps are handled in US/Eastern timezone. VWAP resets at 9:30 ET (RTH open). Futures session boundary at 18:00 ET
- All positions flatten by 15:55 ET (time stop)
- Trading window: 10:00-14:00 ET (no entries outside this window)
- Backtest cost model: commission $0.62/side + 0.25 ticks slippage per side for MES
- VIX regime gates: EXTREME (>35) blocks Strategy A entries entirely, HIGH (25-35) widens BB bands and cuts position size 50%
- Research notebooks in `notebooks/` are `.py` files (not `.ipynb`), likely used with VS Code's interactive Python/Jupytext
