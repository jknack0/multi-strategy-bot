"""
Tests for data.supabase_client.SupabaseClient.

All tests use mocked Supabase — no live connection needed.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs):
    """Create a SupabaseClient with a mocked create_client."""
    with patch("data.supabase_client.create_client") as mock_create:
        mock_sb = MagicMock()
        mock_create.return_value = mock_sb
        from data.supabase_client import SupabaseClient

        client = SupabaseClient(
            url=kwargs.get("url", "https://test.supabase.co"),
            key=kwargs.get("key", "test-key"),
            batch_size=kwargs.get("batch_size", 1000),
        )
        return client, mock_sb


def _mock_table_chain(mock_sb, table_name):
    """Return the mock table object for chaining assertions."""
    return mock_sb.table(table_name)


def _sample_df(n=5, has_vwap=True):
    """Build a small OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min")
    data = {
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000 + i * 10 for i in range(n)],
    }
    if has_vwap:
        data["vwap"] = [100.25 + i for i in range(n)]
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="SUPABASE_URL"):
            with patch("data.supabase_client.create_client"):
                from data.supabase_client import SupabaseClient

                SupabaseClient(url="", key="some-key")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="SUPABASE_KEY"):
            with patch("data.supabase_client.create_client"):
                from data.supabase_client import SupabaseClient

                SupabaseClient(url="https://test.supabase.co", key="")

    def test_valid_init(self):
        client, mock_sb = _make_client()
        assert client.url == "https://test.supabase.co"
        assert client.batch_size == 1000


# ---------------------------------------------------------------------------
# TestMarketDataUpsert
# ---------------------------------------------------------------------------

class TestMarketDataUpsert:
    def test_correct_dict_format(self):
        client, mock_sb = _make_client()
        df = _sample_df(n=1)

        # Set up chain mock
        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        client.upsert_market_data(df, symbol="MES", timeframe="1m")

        # Check what was upserted
        call_args = table_mock.upsert.call_args
        rows = call_args[0][0]
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "MES"
        assert row["timeframe"] == "1m"
        assert row["source"] == "databento"
        assert "open_time" in row
        assert isinstance(row["open"], float)
        assert isinstance(row["volume"], float)  # schema: double precision
        assert "vwap" in row

    def test_batching_splits_correctly(self):
        client, mock_sb = _make_client(batch_size=1000)
        df = _sample_df(n=2500)

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        result = client.upsert_market_data(df, symbol="MES", timeframe="1m")
        assert result == 2500
        # 2500 rows / 1000 batch = 3 calls
        assert table_mock.upsert.call_count == 3

    def test_no_vwap_column(self):
        client, mock_sb = _make_client()
        df = _sample_df(n=1, has_vwap=False)

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        client.upsert_market_data(df, symbol="MES", timeframe="1m")
        rows = table_mock.upsert.call_args[0][0]
        assert "vwap" not in rows[0]

    def test_timestamps_are_iso_strings(self):
        client, mock_sb = _make_client()
        df = _sample_df(n=1)

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        client.upsert_market_data(df, symbol="MES", timeframe="1m")
        rows = table_mock.upsert.call_args[0][0]
        # ISO string should contain 'T'
        assert "T" in rows[0]["open_time"]

    def test_return_count(self):
        client, mock_sb = _make_client()
        df = _sample_df(n=3)

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        result = client.upsert_market_data(df, symbol="MES", timeframe="1m")
        assert result == 3


# ---------------------------------------------------------------------------
# TestRoundTrips
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def _make_trade_record(self):
        """Minimal TradeRecord-like object."""
        from dataclasses import dataclass

        @dataclass
        class FakeTradeRecord:
            side: str
            entry_price: float
            exit_price: float
            entry_time: datetime
            exit_time: datetime
            contracts: int
            pnl: float
            exit_reason: str

        return FakeTradeRecord(
            side="LONG",
            entry_price=4500.0,
            exit_price=4505.0,
            entry_time=datetime(2024, 1, 2, 10, 30),
            exit_time=datetime(2024, 1, 2, 11, 15),
            contracts=1,
            pnl=25.0,
            exit_reason="take_profit",
        )

    def _make_orb_trade_record(self):
        """Minimal ORBTradeRecord-like object with or_width."""
        from dataclasses import dataclass

        @dataclass
        class FakeORBTradeRecord:
            side: str
            entry_price: float
            exit_price: float
            entry_time: datetime
            exit_time: datetime
            contracts: int
            pnl: float
            exit_reason: str
            or_width: float
            or_duration_minutes: int

        return FakeORBTradeRecord(
            side="LONG",
            entry_price=4500.0,
            exit_price=4510.0,
            entry_time=datetime(2024, 1, 2, 10, 0),
            exit_time=datetime(2024, 1, 2, 10, 45),
            contracts=2,
            pnl=100.0,
            exit_reason="target",
            or_width=5.0,
            or_duration_minutes=15,
        )

    def test_trade_record_mapping(self):
        client, mock_sb = _make_client()
        trade = self._make_trade_record()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        insert_mock = MagicMock()
        table_mock.insert.return_value = insert_mock
        insert_mock.execute.return_value = MagicMock(data=[{"id": "trade-1"}])

        result = client.insert_trades_from_records([trade], "strat-a", "MES")
        assert result == 1

        # Verify insert calls: 2 trades (entry buy + exit sell) + 1 round_trip batch
        assert len(table_mock.insert.call_args_list) == 3

        # Check entry trade uses schema fields
        entry_row = table_mock.insert.call_args_list[0][0][0]
        assert entry_row["side"] == "buy"  # LONG entry → buy
        assert entry_row["executed_at"]  # not 'timestamp'
        assert entry_row["broker"] == "ib"
        assert "commission" in entry_row
        assert "trade_type" not in entry_row

        # Check exit trade
        exit_row = table_mock.insert.call_args_list[1][0][0]
        assert exit_row["side"] == "sell"  # LONG exit → sell

    def test_orb_includes_metadata(self):
        client, mock_sb = _make_client()
        trade = self._make_orb_trade_record()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        insert_mock = MagicMock()
        table_mock.insert.return_value = insert_mock
        insert_mock.execute.return_value = MagicMock(data=[{"id": "trade-1"}])

        client.insert_trades_from_records([trade], "strat-b", "MES")

        # Find the round_trips insert call (last insert)
        last_call = table_mock.insert.call_args_list[-1]
        rt_rows = last_call[0][0]
        assert len(rt_rows) == 1
        # ORB fields in metadata jsonb
        assert rt_rows[0]["metadata"]["or_width"] == 5.0
        assert rt_rows[0]["metadata"]["or_duration_minutes"] == 15
        # exit_reason is a top-level column, not in metadata
        assert rt_rows[0]["exit_reason"] == "target"
        assert "exit_reason" not in rt_rows[0]["metadata"]

    def test_net_pnl_subtracts_commission(self):
        client, mock_sb = _make_client()
        trade = self._make_trade_record()  # contracts=1, pnl=25.0

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        insert_mock = MagicMock()
        table_mock.insert.return_value = insert_mock
        insert_mock.execute.return_value = MagicMock(data=[{"id": "trade-1"}])

        client.insert_trades_from_records(
            [trade], "strat-a", "MES", commission_per_side=0.62
        )

        last_call = table_mock.insert.call_args_list[-1]
        rt_rows = last_call[0][0]
        expected_net = 25.0 - (0.62 * 2 * 1)  # gross - 2*comm*contracts
        assert rt_rows[0]["net_pnl"] == pytest.approx(expected_net)

    def test_hold_duration_computed(self):
        client, mock_sb = _make_client()
        trade = self._make_trade_record()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        insert_mock = MagicMock()
        table_mock.insert.return_value = insert_mock
        insert_mock.execute.return_value = MagicMock(data=[{"id": "trade-1"}])

        client.insert_trades_from_records([trade], "strat-a", "MES")

        last_call = table_mock.insert.call_args_list[-1]
        rt_rows = last_call[0][0]
        expected_duration = str(
            datetime(2024, 1, 2, 11, 15) - datetime(2024, 1, 2, 10, 30)
        )
        assert rt_rows[0]["hold_duration"] == expected_duration

    def test_round_trip_uses_schema_columns(self):
        client, mock_sb = _make_client()
        trade = self._make_trade_record()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        insert_mock = MagicMock()
        table_mock.insert.return_value = insert_mock
        insert_mock.execute.return_value = MagicMock(data=[{"id": "trade-1"}])

        client.insert_trades_from_records([trade], "strat-a", "MES")

        last_call = table_mock.insert.call_args_list[-1]
        rt = last_call[0][0][0]
        # Schema uses direction/opened_at/closed_at, not side/entry_time/exit_time
        assert rt["direction"] == "long"
        assert "opened_at" in rt
        assert "closed_at" in rt
        assert "side" not in rt
        assert "entry_time" not in rt
        assert "exit_time" not in rt


# ---------------------------------------------------------------------------
# TestSnapshots
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_metrics_mapping(self):
        client, mock_sb = _make_client()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        upsert_mock = MagicMock()
        table_mock.upsert.return_value = upsert_mock
        upsert_mock.execute.return_value = MagicMock(data=[])

        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.05,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "total_trades": 100,
            "expectancy": 12.5,
            "total_return": 0.15,
            "calmar": 3.0,
            "max_drawdown_duration": 45,
        }

        client.insert_snapshot("strat-a", "2024-01-31", "MES", metrics)

        call_args = table_mock.upsert.call_args[0][0]
        assert call_args["sharpe"] == 1.5
        assert call_args["sortino"] == 2.0
        assert call_args["max_drawdown"] == 0.05
        assert call_args["profit_factor"] == 1.8
        assert call_args["win_rate"] == 0.55
        assert call_args["total_trades"] == 100
        assert call_args["avg_pnl"] == 12.5  # expectancy → avg_pnl
        assert call_args["metadata"]["total_return"] == 0.15
        assert call_args["metadata"]["calmar"] == 3.0


# ---------------------------------------------------------------------------
# TestBatching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_exact_division(self):
        from data.supabase_client import SupabaseClient

        chunks = list(SupabaseClient._chunked([1, 2, 3, 4], 2))
        assert chunks == [[1, 2], [3, 4]]

    def test_remainder(self):
        from data.supabase_client import SupabaseClient

        chunks = list(SupabaseClient._chunked([1, 2, 3, 4, 5], 2))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_empty_list(self):
        from data.supabase_client import SupabaseClient

        chunks = list(SupabaseClient._chunked([], 10))
        assert chunks == []

    def test_single_chunk(self):
        from data.supabase_client import SupabaseClient

        chunks = list(SupabaseClient._chunked([1, 2, 3], 10))
        assert chunks == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_success(self):
        client, mock_sb = _make_client()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        select_mock = MagicMock()
        table_mock.select.return_value = select_mock
        limit_mock = MagicMock()
        select_mock.limit.return_value = limit_mock
        limit_mock.execute.return_value = MagicMock(data=[])

        assert client.health_check() is True

    def test_failure(self):
        client, mock_sb = _make_client()

        table_mock = MagicMock()
        mock_sb.table.return_value = table_mock
        table_mock.select.side_effect = Exception("connection refused")

        assert client.health_check() is False
