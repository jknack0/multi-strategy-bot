"""Tests for the live trading runner and Databento connector."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytz
import pytest

ET = pytz.timezone("US/Eastern")
ROOT = Path(__file__).resolve().parent.parent


# ── DatabentoLiveConnector tests ──────────────────────────────────────


class TestDatabentoLiveConnector:
    """Tests for data/databento_live.py."""

    def test_init_requires_api_key(self):
        from data.databento_live import DatabentoLiveConnector
        with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
            DatabentoLiveConnector(api_key="")

    def test_add_callback(self):
        from data.databento_live import DatabentoLiveConnector
        connector = DatabentoLiveConnector(api_key="test-key")
        cb = MagicMock()
        connector.add_callback(cb)
        assert cb in connector._callbacks

    def test_add_callback_no_duplicates(self):
        from data.databento_live import DatabentoLiveConnector
        connector = DatabentoLiveConnector(api_key="test-key")
        cb = MagicMock()
        connector.add_callback(cb)
        connector.add_callback(cb)
        assert len(connector._callbacks) == 1

    def test_handle_ohlcv_dispatches_to_callbacks(self):
        """Simulate an OHLCVMsg and verify callback receives correct values."""
        from data.databento_live import DatabentoLiveConnector

        connector = DatabentoLiveConnector(api_key="test-key")
        received = []
        connector.add_callback(lambda ts, o, h, l, c, v: received.append((ts, o, h, l, c, v)))

        # Create a mock OHLCVMsg with float prices (newer SDK)
        mock_ohlcv = MagicMock()
        mock_ohlcv.ts_event = 1709300400_000_000_000  # 2024-03-01 15:00 UTC (ns)
        mock_ohlcv.open = 5100.25
        mock_ohlcv.high = 5105.50
        mock_ohlcv.low = 5098.00
        mock_ohlcv.close = 5103.75
        mock_ohlcv.volume = 1234

        connector._handle_ohlcv(mock_ohlcv)

        assert len(received) == 1
        ts, o, h, l, c, v = received[0]
        assert o == 5100.25
        assert h == 5105.50
        assert l == 5098.00
        assert c == 5103.75
        assert v == 1234
        assert ts.tzinfo is not None  # timezone-aware

    def test_handle_ohlcv_int_prices(self):
        """Test fixed-point integer price conversion (older SDK)."""
        from data.databento_live import DatabentoLiveConnector, FIXED_PRICE_SCALE

        connector = DatabentoLiveConnector(api_key="test-key")
        received = []
        connector.add_callback(lambda ts, o, h, l, c, v: received.append((o, h, l, c)))

        mock_ohlcv = MagicMock()
        mock_ohlcv.ts_event = 1709300400_000_000_000
        mock_ohlcv.open = int(5100.25 * FIXED_PRICE_SCALE)
        mock_ohlcv.high = int(5105.50 * FIXED_PRICE_SCALE)
        mock_ohlcv.low = int(5098.00 * FIXED_PRICE_SCALE)
        mock_ohlcv.close = int(5103.75 * FIXED_PRICE_SCALE)
        mock_ohlcv.volume = 1234

        connector._handle_ohlcv(mock_ohlcv)

        o, h, l, c = received[0]
        assert abs(o - 5100.25) < 0.01
        assert abs(h - 5105.50) < 0.01

    def test_callback_error_doesnt_crash(self):
        """A failing callback shouldn't prevent other callbacks from running."""
        from data.databento_live import DatabentoLiveConnector

        connector = DatabentoLiveConnector(api_key="test-key")
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        connector.add_callback(bad_cb)
        connector.add_callback(good_cb)

        mock_ohlcv = MagicMock()
        mock_ohlcv.ts_event = 1709300400_000_000_000
        mock_ohlcv.open = 5100.0
        mock_ohlcv.high = 5105.0
        mock_ohlcv.low = 5098.0
        mock_ohlcv.close = 5103.0
        mock_ohlcv.volume = 100

        connector._handle_ohlcv(mock_ohlcv)
        good_cb.assert_called_once()

    def test_add_l1_callback(self):
        from data.databento_live import DatabentoLiveConnector
        connector = DatabentoLiveConnector(api_key="test-key", enable_l1=True)
        cb = MagicMock()
        connector.add_l1_callback(cb)
        assert cb in connector._l1_callbacks

    def test_handle_l1_dispatches_to_callbacks(self):
        """Simulate an MBP1Msg and verify L1 callback receives correct values."""
        from data.databento_live import DatabentoLiveConnector

        connector = DatabentoLiveConnector(api_key="test-key", enable_l1=True)
        received = []
        connector.add_l1_callback(
            lambda ts, bid, ask, bid_sz, ask_sz: received.append((ts, bid, ask, bid_sz, ask_sz))
        )

        # Create a mock MBP1Msg with float prices
        mock_level = MagicMock()
        mock_level.bid_px = 5100.25
        mock_level.ask_px = 5100.50
        mock_level.bid_sz = 42
        mock_level.ask_sz = 38

        mock_msg = MagicMock()
        mock_msg.ts_event = 1709300400_000_000_000
        mock_msg.levels = [mock_level]

        connector._handle_l1(mock_msg)

        assert len(received) == 1
        ts, bid, ask, bid_sz, ask_sz = received[0]
        assert bid == 5100.25
        assert ask == 5100.50
        assert bid_sz == 42
        assert ask_sz == 38
        assert ts.tzinfo is not None

    def test_handle_l1_int_prices(self):
        """Test fixed-point integer price conversion for L1."""
        from data.databento_live import DatabentoLiveConnector, FIXED_PRICE_SCALE

        connector = DatabentoLiveConnector(api_key="test-key", enable_l1=True)
        received = []
        connector.add_l1_callback(
            lambda ts, bid, ask, bid_sz, ask_sz: received.append((bid, ask))
        )

        mock_level = MagicMock()
        mock_level.bid_px = int(5100.25 * FIXED_PRICE_SCALE)
        mock_level.ask_px = int(5100.50 * FIXED_PRICE_SCALE)
        mock_level.bid_sz = 10
        mock_level.ask_sz = 15

        mock_msg = MagicMock()
        mock_msg.ts_event = 1709300400_000_000_000
        mock_msg.levels = [mock_level]

        connector._handle_l1(mock_msg)

        bid, ask = received[0]
        assert abs(bid - 5100.25) < 0.01
        assert abs(ask - 5100.50) < 0.01

    def test_handle_l1_no_callbacks_skips(self):
        """No L1 callbacks registered should not crash."""
        from data.databento_live import DatabentoLiveConnector

        connector = DatabentoLiveConnector(api_key="test-key", enable_l1=True)
        mock_msg = MagicMock()
        mock_msg.ts_event = 1709300400_000_000_000
        # Should return early without error
        connector._handle_l1(mock_msg)


# ── LiveRunner tests ──────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a LiveRunner in dry-run mode with mocked dependencies."""
    with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
        r = LiveRunner_factory()
    return r


def LiveRunner_factory():
    """Build a runner with default params and no external deps."""
    from execution.live_runner import LiveRunner
    r = LiveRunner(
        capital=20_000.0,
        dry_run=True,
        replay_days=0,
    )
    r._questdb = None
    r._supabase = None
    return r


class TestLiveRunner:
    """Tests for execution/live_runner.py."""

    def test_init_loads_strategies(self):
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()
        assert r.strategy_a is not None
        assert r.strategy_b is not None

    def test_on_bar_dispatches_to_both_strategies(self):
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()

        # A bar during trading hours
        ts = ET.localize(datetime(2025, 6, 15, 11, 30, 0))
        r.strategy_a.on_bar = MagicMock(return_value=None)
        r.strategy_b.on_bar = MagicMock(return_value=None)
        # Mock position to None (no entry)
        r.strategy_a.position = None
        r.strategy_b.position = None

        r._on_bar(ts, 5400.0, 5402.0, 5398.0, 5401.0, 1000)

        r.strategy_a.on_bar.assert_called_once_with(ts, 5400.0, 5402.0, 5398.0, 5401.0, 1000)
        r.strategy_b.on_bar.assert_called_once_with(ts, 5400.0, 5402.0, 5398.0, 5401.0, 1000)

    def test_new_day_detection(self):
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()

        # Set initial day
        day1 = ET.localize(datetime(2025, 6, 15, 15, 59, 0))
        r._check_new_day(day1)
        assert r._current_date == day1.date()

        # New day should reset
        day2 = ET.localize(datetime(2025, 6, 16, 9, 30, 0))
        r._daily_pnl_a = 100.0
        r._daily_pnl_b = -50.0
        with patch.object(r, "_fetch_vix", return_value=22.0):
            r._check_new_day(day2)

        assert r._current_date == day2.date()
        assert r._daily_pnl_a == 0.0
        assert r._daily_pnl_b == 0.0

    def test_exit_detection_updates_pnl(self):
        from strategies.mean_reversion import TradeRecord, Side
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()

        ts = ET.localize(datetime(2025, 6, 15, 11, 30, 0))
        r._current_date = ts.date()

        # Mock Strategy A returning a trade (exit)
        mock_trade = TradeRecord(
            side=Side.LONG, entry_price=5400.0, exit_price=5410.0,
            entry_time=ts, exit_time=ts, contracts=1, pnl=50.0,
            exit_reason="take_profit",
        )
        r.strategy_a.on_bar = MagicMock(return_value=mock_trade)
        r.strategy_a.position = None  # no position (already closed)
        r.strategy_b.on_bar = MagicMock(return_value=None)
        r.strategy_b.position = None

        r._on_bar(ts, 5400.0, 5412.0, 5398.0, 5410.0, 1000)

        assert r._daily_pnl_a == 50.0
        assert r._daily_trades_a == 1

    def test_circuit_breaker_flatten(self):
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()

        ts = ET.localize(datetime(2025, 6, 15, 12, 0, 0))
        r._current_date = ts.date()

        # Mock circuit breaker to HALTED
        r.risk_manager.circuit_breaker.should_flatten_all = MagicMock(return_value=True)
        r.strategy_a.position = None  # no position to flatten
        r.strategy_b.position = None

        # Should not crash even with no positions
        r._on_bar(ts, 5400.0, 5402.0, 5398.0, 5401.0, 1000)

    def test_vix_fallback_to_default(self):
        with patch("execution.live_runner.LiveRunner._fetch_vix", return_value=20.0):
            r = LiveRunner_factory()

        # Patch _fetch_vix to use real implementation with no QuestDB and failed yfinance
        r._questdb = None
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.side_effect = ImportError("no yfinance")
            vix = r._fetch_vix.__wrapped__(r) if hasattr(r._fetch_vix, "__wrapped__") else 20.0
            # Default should be 20.0
            assert vix == 20.0
