"""Minimal Streamlit app: ticker lookup with DuckDB cache + yfinance fetch."""

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "stocks.duckdb"
# Treat cached data as fresh if the latest bar is within this many calendar days.
FRESHNESS_DAYS = 3
HISTORY_PERIOD = "1y"


def get_connection() -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    return conn


def has_recent_data(conn: duckdb.DuckDBPyConnection, ticker: str) -> bool:
    row = conn.execute(
        """
        SELECT MAX(date) AS latest
        FROM daily_prices
        WHERE ticker = ?
        """,
        [ticker],
    ).fetchone()
    if row is None or row[0] is None:
        return False
    latest: date = row[0]
    return latest >= date.today() - timedelta(days=FRESHNESS_DAYS)


def load_cached(conn: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT date, open, high, low, close, volume
        FROM daily_prices
        WHERE ticker = ?
        ORDER BY date
        """,
        [ticker],
    ).df()


def fetch_and_store(conn: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance may return MultiIndex columns when downloading a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df = df.rename(columns=str.lower)
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker
    df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]

    conn.execute("DELETE FROM daily_prices WHERE ticker = ?", [ticker])
    conn.register("_incoming", df)
    conn.execute(
        """
        INSERT INTO daily_prices
        SELECT ticker, date, open, high, low, close, CAST(volume AS BIGINT)
        FROM _incoming
        """
    )
    conn.unregister("_incoming")
    return df.drop(columns=["ticker"])


def get_prices(conn: duckdb.DuckDBPyConnection, ticker: str) -> tuple[pd.DataFrame, str]:
    if has_recent_data(conn, ticker):
        return load_cached(conn, ticker), "cache"
    return fetch_and_store(conn, ticker), "yfinance"


st.set_page_config(page_title="Stock Lenses", layout="wide")
st.title("Stock Lenses")
st.caption("Type a ticker. Prices are cached in DuckDB; fresh data is pulled from yfinance.")

with st.form("ticker_form"):
    ticker_input = st.text_input("Ticker", value="AAPL").strip().upper()
    submitted = st.form_submit_button("Load", type="primary")

if submitted:
    if not ticker_input:
        st.warning("Enter a ticker symbol.")
    else:
        try:
            with get_connection() as conn:
                with st.spinner(f"Loading {ticker_input}…"):
                    prices, source = get_prices(conn, ticker_input)

            if prices.empty:
                st.error(f"No prices available for {ticker_input}.")
            else:
                label = "DuckDB cache" if source == "cache" else "yfinance (fetched & stored)"
                st.success(f"{ticker_input}: {len(prices)} daily bars from {label}")
                fig = px.line(
                    prices,
                    x="date",
                    y="close",
                    title=f"{ticker_input} close price",
                    labels={"date": "Date", "close": "Close"},
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Raw data"):
                    st.dataframe(prices, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 — surface fetch/parse errors in the UI
            st.error(str(exc))
