"""
Supabase client for the analytics/history warehouse layer.

Handles upserting market data, trade records, and performance snapshots
to the BayesBot Supabase schema. This is a write-mostly analytics layer;
QuestDB remains the local hot-path store.

Tables used: market_data, strategies, trades, round_trips, strategy_snapshots
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase import create_client, Client

from config.settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_BATCH_SIZE

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for Supabase analytics warehouse."""

    def __init__(
        self,
        url: str = SUPABASE_URL,
        key: str = SUPABASE_KEY,
        batch_size: int = SUPABASE_BATCH_SIZE,
    ) -> None:
        if not url:
            raise ValueError("SUPABASE_URL is required")
        if not key:
            raise ValueError("SUPABASE_KEY is required")
        self.url = url
        self.key = key
        self.batch_size = batch_size
        self.client: Client = create_client(url, key)

    # ── Market Data ───────────────────────────────────────────────────────

    def upsert_market_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str = "databento",
    ) -> int:
        """Batch upsert OHLCV bars from a DataFrame into market_data.

        Expects df with DatetimeIndex and columns: open, high, low, close, volume.
        Optional column: vwap.

        Returns the total number of rows upserted.
        """
        records = []
        for ts, row in df.iterrows():
            record: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": _ts_to_iso(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "source": source,
            }
            if "vwap" in row.index and pd.notna(row["vwap"]):
                record["vwap"] = float(row["vwap"])
            records.append(record)

        total = 0
        for chunk in self._chunked(records, self.batch_size):
            self.client.table("market_data").upsert(
                chunk, on_conflict="symbol,timeframe,open_time"
            ).execute()
            total += len(chunk)
            logger.info("Upserted %d / %d market_data rows", total, len(records))

        return total

    def upsert_bar(
        self,
        symbol: str,
        timeframe: str,
        open_time: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        source: str = "databento",
    ) -> None:
        """Upsert a single OHLCV bar into market_data."""
        record = {
            "symbol": symbol,
            "timeframe": timeframe,
            "open_time": _ts_to_iso(open_time),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "source": source,
        }
        self.client.table("market_data").upsert(
            record, on_conflict="symbol,timeframe,open_time"
        ).execute()

    def insert_l1_batch(
        self,
        records: list,
    ) -> int:
        """Batch insert L1 tick records into market_data_l1.

        Each record should be a dict with keys:
            symbol, ts, bid, ask, bid_size, ask_size, source
        Returns the number of rows inserted.
        """
        total = 0
        for chunk in self._chunked(records, self.batch_size):
            self.client.table("market_data_l1").insert(chunk).execute()
            total += len(chunk)
        return total

    def get_latest_market_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[datetime]:
        """Return the latest open_time for a symbol/timeframe, or None."""
        resp = (
            self.client.table("market_data")
            .select("open_time")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .order("open_time", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return pd.to_datetime(resp.data[0]["open_time"]).to_pydatetime()
        return None

    def count_market_data(self, symbol: str, timeframe: str) -> int:
        """Return the row count for a symbol/timeframe."""
        resp = (
            self.client.table("market_data")
            .select("*", count="exact")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .execute()
        )
        return resp.count or 0

    # ── Strategies ────────────────────────────────────────────────────────

    def upsert_strategy(
        self,
        strategy_id: str,
        name: str,
        version: str,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        asset_class: str = "futures",
        status: str = "active",
    ) -> None:
        """Register or update a strategy in the strategies table."""
        record = {
            "id": strategy_id,
            "name": name,
            "version": version,
            "config": config or {},
            "description": description,
            "asset_class": asset_class,
            "status": status,
        }
        self.client.table("strategies").upsert(record, on_conflict="id").execute()
        logger.info("Upserted strategy %s (%s v%s)", strategy_id, name, version)

    # ── Trades & Round Trips ──────────────────────────────────────────────

    def insert_trades_from_records(
        self,
        trade_records: list,
        strategy_id: str,
        symbol: str,
        commission_per_side: float = 0.62,
        broker: str = "ib",
    ) -> int:
        """Convert TradeRecord/ORBTradeRecord dataclasses into trades + round_trips rows.

        Returns the number of round_trips inserted.
        """
        round_trip_rows = []

        for trade in trade_records:
            direction = str(trade.side).lower()  # 'long' or 'short'
            if hasattr(trade.side, "value"):
                direction = trade.side.value.lower()
            entry_time_iso = _ts_to_iso(trade.entry_time)
            exit_time_iso = _ts_to_iso(trade.exit_time)
            total_commission = commission_per_side * 2 * trade.contracts
            per_leg_commission = commission_per_side * trade.contracts

            # Map direction to buy/sell for trades table
            entry_side = "buy" if direction == "long" else "sell"
            exit_side = "sell" if direction == "long" else "buy"

            # Insert entry trade
            entry_row = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": entry_side,
                "price": trade.entry_price,
                "quantity": trade.contracts,
                "commission": per_leg_commission,
                "executed_at": entry_time_iso,
                "broker": broker,
            }
            entry_resp = (
                self.client.table("trades").insert(entry_row).execute()
            )
            entry_trade_id = entry_resp.data[0]["id"] if entry_resp.data else None

            # Insert exit trade
            exit_row = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": exit_side,
                "price": trade.exit_price,
                "quantity": trade.contracts,
                "commission": per_leg_commission,
                "executed_at": exit_time_iso,
                "broker": broker,
            }
            exit_resp = (
                self.client.table("trades").insert(exit_row).execute()
            )
            exit_trade_id = exit_resp.data[0]["id"] if exit_resp.data else None

            # Compute round trip fields
            gross_pnl = trade.pnl
            net_pnl = gross_pnl - total_commission
            hold_duration = str(trade.exit_time - trade.entry_time)

            # Build metadata (ORB-specific fields)
            metadata: Dict[str, Any] = {}
            if hasattr(trade, "or_width"):
                metadata["or_width"] = trade.or_width
                metadata["or_duration_minutes"] = trade.or_duration_minutes

            rt_row: Dict[str, Any] = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "direction": direction,
                "entry_trade_id": entry_trade_id,
                "exit_trade_id": exit_trade_id,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.contracts,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "hold_duration": hold_duration,
                "exit_reason": trade.exit_reason,
                "opened_at": entry_time_iso,
                "closed_at": exit_time_iso,
                "metadata": metadata,
            }
            round_trip_rows.append(rt_row)

        # Batch insert round trips
        total = 0
        for chunk in self._chunked(round_trip_rows, self.batch_size):
            self.client.table("round_trips").insert(chunk).execute()
            total += len(chunk)

        logger.info("Inserted %d round_trips for strategy %s", total, strategy_id)
        return total

    # ── Strategy Snapshots ────────────────────────────────────────────────

    def insert_snapshot(
        self,
        strategy_id: str,
        snapshot_date: str,
        symbol: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Write a performance metrics snapshot.

        Maps compute_all_metrics() keys to strategy_snapshots columns.
        """
        record: Dict[str, Any] = {
            "strategy_id": strategy_id,
            "snapshot_date": snapshot_date,
            "symbol": symbol,
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "max_drawdown": metrics.get("max_drawdown"),
            "profit_factor": metrics.get("profit_factor"),
            "win_rate": metrics.get("win_rate"),
            "total_trades": metrics.get("total_trades"),
            "avg_pnl": metrics.get("expectancy"),
            "metadata": {
                "total_return": metrics.get("total_return"),
                "calmar": metrics.get("calmar"),
                "max_drawdown_duration": metrics.get("max_drawdown_duration"),
            },
        }
        self.client.table("strategy_snapshots").upsert(
            record, on_conflict="strategy_id,snapshot_date,symbol"
        ).execute()
        logger.info(
            "Upserted snapshot for %s on %s (%s)", strategy_id, snapshot_date, symbol
        )

    # ── Health Check ──────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Verify connectivity with a simple select. Returns True if healthy."""
        try:
            self.client.table("strategies").select("id").limit(1).execute()
            return True
        except Exception as exc:
            logger.warning("Supabase health check failed: %s", exc)
            return False

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _chunked(lst: list, size: int) -> list:
        """Yield successive chunks of `size` from `lst`."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


def _ts_to_iso(ts) -> str:
    """Convert a timestamp (datetime or pd.Timestamp) to ISO 8601 string."""
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)
