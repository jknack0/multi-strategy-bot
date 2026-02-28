"""
QuestDB client for ingesting OHLCV bars via InfluxDB Line Protocol
and querying via PostgreSQL wire protocol.

Schema:
    ohlcv(
        timestamp TIMESTAMP,
        symbol SYMBOL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume LONG,
        vwap DOUBLE,
        bar_size SYMBOL
    ) TIMESTAMP(timestamp) PARTITION BY MONTH
"""

import socket
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import (
    QUESTDB_HOST,
    QUESTDB_ILP_PORT,
    QUESTDB_PG_PORT,
    QUESTDB_PG_USER,
    QUESTDB_PG_PASSWORD,
    QUESTDB_PG_DATABASE,
)

logger = logging.getLogger(__name__)


class QuestDBClient:
    """Client for QuestDB ingestion and queries."""

    TABLE_NAME: str = "ohlcv"

    def __init__(
        self,
        host: str = QUESTDB_HOST,
        ilp_port: int = QUESTDB_ILP_PORT,
        pg_port: int = QUESTDB_PG_PORT,
        pg_user: str = QUESTDB_PG_USER,
        pg_password: str = QUESTDB_PG_PASSWORD,
        pg_database: str = QUESTDB_PG_DATABASE,
    ) -> None:
        self.host = host
        self.ilp_port = ilp_port
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_database = pg_database
        self._ilp_socket: Optional[socket.socket] = None

    # ── ILP Ingestion ────────────────────────────────────────────────────

    def _connect_ilp(self) -> socket.socket:
        """Open a TCP socket to QuestDB's ILP endpoint."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.ilp_port))
        logger.info("Connected to QuestDB ILP at %s:%d", self.host, self.ilp_port)
        return sock

    def _ensure_ilp_socket(self) -> socket.socket:
        if self._ilp_socket is None:
            self._ilp_socket = self._connect_ilp()
        return self._ilp_socket

    @staticmethod
    def _escape_tag(value: str) -> str:
        """Escape special chars for ILP tag values."""
        return value.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    @staticmethod
    def _to_epoch_ns(ts: datetime) -> int:
        """Convert a datetime to nanosecond epoch."""
        return int(ts.timestamp() * 1_000_000_000)

    def build_ilp_line(
        self,
        symbol: str,
        bar_size: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap: float = 0.0,
    ) -> str:
        """Build a single InfluxDB Line Protocol string for an OHLCV bar."""
        sym = self._escape_tag(symbol)
        bs = self._escape_tag(bar_size)
        ts_ns = self._to_epoch_ns(timestamp)
        line = (
            f"{self.TABLE_NAME},symbol={sym},bar_size={bs} "
            f"open={open_},high={high},low={low},close={close},"
            f"volume={volume}i,vwap={vwap} "
            f"{ts_ns}\n"
        )
        return line

    def ingest_bar(
        self,
        symbol: str,
        bar_size: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap: float = 0.0,
    ) -> None:
        """Send a single OHLCV bar to QuestDB via ILP."""
        sock = self._ensure_ilp_socket()
        line = self.build_ilp_line(
            symbol, bar_size, timestamp, open_, high, low, close, volume, vwap
        )
        sock.sendall(line.encode())

    def ingest_bars_batch(self, lines: List[str]) -> None:
        """Send a batch of pre-built ILP lines to QuestDB."""
        if not lines:
            return
        sock = self._ensure_ilp_socket()
        payload = "".join(lines)
        sock.sendall(payload.encode())
        logger.info("Ingested batch of %d bars", len(lines))

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        bar_size: str,
    ) -> None:
        """Ingest a DataFrame with columns [timestamp, open, high, low, close, volume]
        and optional [vwap] into QuestDB."""
        lines: List[str] = []
        for _, row in df.iterrows():
            ts = row["timestamp"] if isinstance(row["timestamp"], datetime) else pd.Timestamp(row["timestamp"]).to_pydatetime()
            vwap_val = row.get("vwap", 0.0)
            if pd.isna(vwap_val):
                vwap_val = 0.0
            lines.append(
                self.build_ilp_line(
                    symbol=symbol,
                    bar_size=bar_size,
                    timestamp=ts,
                    open_=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    vwap=float(vwap_val),
                )
            )
            # Flush in chunks of 1000
            if len(lines) >= 1000:
                self.ingest_bars_batch(lines)
                lines.clear()
        if lines:
            self.ingest_bars_batch(lines)

    def close_ilp(self) -> None:
        """Close the ILP socket."""
        if self._ilp_socket is not None:
            self._ilp_socket.close()
            self._ilp_socket = None
            logger.info("Closed QuestDB ILP socket")

    # ── PostgreSQL Queries ───────────────────────────────────────────────

    def _pg_connection(self):
        """Create a new PostgreSQL connection to QuestDB."""
        return psycopg2.connect(
            host=self.host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_database,
        )

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""
        conn = self._pg_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows)
        finally:
            conn.close()

    def get_bars(
        self,
        symbol: str,
        bar_size: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from QuestDB, returned as a pandas DataFrame."""
        conditions = [f"symbol = '{symbol}'", f"bar_size = '{bar_size}'"]
        if start:
            conditions.append(f"timestamp >= '{start.isoformat()}'")
        if end:
            conditions.append(f"timestamp <= '{end.isoformat()}'")
        where = " AND ".join(conditions)
        sql = (
            f"SELECT timestamp, symbol, open, high, low, close, volume, vwap, bar_size "
            f"FROM {self.TABLE_NAME} "
            f"WHERE {where} "
            f"ORDER BY timestamp ASC "
            f"LIMIT {limit}"
        )
        df = self.query(sql)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        return df

    def create_table(self) -> None:
        """Create the OHLCV table if it doesn't exist (via REST/SQL exec)."""
        import requests
        sql = (
            f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} ("
            "timestamp TIMESTAMP, "
            "symbol SYMBOL, "
            "open DOUBLE, "
            "high DOUBLE, "
            "low DOUBLE, "
            "close DOUBLE, "
            "volume LONG, "
            "vwap DOUBLE, "
            "bar_size SYMBOL"
            ") TIMESTAMP(timestamp) PARTITION BY MONTH;"
        )
        url = f"http://{self.host}:{QUESTDB_HTTP_PORT}/exec"
        try:
            resp = requests.get(url, params={"query": sql}, timeout=10)
            resp.raise_for_status()
            logger.info("Table '%s' created/verified", self.TABLE_NAME)
        except requests.RequestException as exc:
            logger.error("Failed to create table: %s", exc)
            raise


# Module-level convenience import
QUESTDB_HTTP_PORT = int(__import__("os").getenv("QUESTDB_HTTP_PORT", "9000"))
