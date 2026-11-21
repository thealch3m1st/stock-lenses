from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "stocks.duckdb"
FRESHNESS_DAYS = 3
HISTORY_PERIOD = "1y"


@dataclass(frozen=True)
class Basket:
    key: str
    label: str
    tickers: tuple[str, ...]
    start_value: float


BASKETS: dict[str, Basket] = {
    "stocks": Basket(
        key="stocks",
        label="Dollar stocks",
        tickers=("AAPL", "MSFT", "GOOGL", "AMZN"),
        start_value=10_000.0,
    ),
    "crypto": Basket(
        key="crypto",
        label="Crypto",
        tickers=("BTC-USD", "ETH-USD", "SOL-USD"),
        start_value=10_000.0,
    ),
    "metals": Basket(
        key="metals",
        label="Precious metals",
        tickers=("GC=F", "SI=F", "PL=F"),
        start_value=10_000.0,
    ),
}

MOBILE_CSS = """
<style>
.main .block-container {
    max-width: 100%;
    padding-left: 1rem;
    padding-right: 1rem;
}
section.main > div {
    overflow-x: clip;
}
div[data-testid="stMetric"] {
    min-width: 0;
    overflow-wrap: anywhere;
}
@media (max-width: 480px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.25rem 0 !important;
    }
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 0 !important;
    }
    div[data-testid="stPlotlyChart"] {
        overflow-x: hidden;
    }
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plotly,
    div[data-testid="stPlotlyChart"] .main-svg {
        max-width: 100% !important;
    }
}
</style>
"""


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


def align_closes(
    prices_by_ticker: dict[str, pd.DataFrame], tickers: tuple[str, ...]
) -> pd.DataFrame:
    closes: list[pd.Series] = []
    for ticker in tickers:
        frame = prices_by_ticker[ticker]
        if frame.empty or not {"date", "close"}.issubset(frame.columns):
            raise ValueError(f"No prices for {ticker}.")
        cleaned = frame.loc[:, ["date", "close"]].dropna(subset=["close"])
        cleaned = cleaned.drop_duplicates(subset="date", keep="last")
        cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.normalize()
        series = cleaned.set_index("date")["close"].astype(float).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if series.empty:
            raise ValueError(f"No prices for {ticker}.")
        closes.append(series.rename(ticker))

    aligned = pd.concat(closes, axis=1, join="inner").sort_index().dropna(how="any")
    if aligned.empty:
        raise ValueError("These tickers have no dates in common.")
    return aligned


def basket_shares(
    prices_by_ticker: dict[str, pd.DataFrame], basket: Basket
) -> dict[str, float]:
    if basket.start_value <= 0 or not basket.tickers:
        raise ValueError(
            f"{basket.label} needs a positive start value and at least one ticker."
        )
    aligned = align_closes(prices_by_ticker, basket.tickers)
    first = aligned.iloc[0]
    invalid = first.index[first.isna() | (first <= 0)]
    if len(invalid):
        names = ", ".join(str(name) for name in invalid)
        raise ValueError(
            "Cannot start the basket. The first close is missing or not positive"
            f" for {names}."
        )
    per_ticker = (basket.start_value / len(basket.tickers)) / first
    return {str(ticker): float(per_ticker[ticker]) for ticker in basket.tickers}


def basket_history(
    prices_by_ticker: dict[str, pd.DataFrame], basket: Basket
) -> pd.DataFrame:
    if basket.start_value <= 0 or not basket.tickers:
        raise ValueError(
            f"{basket.label} needs a positive start value and at least one ticker."
        )

    aligned = align_closes(prices_by_ticker, basket.tickers)
    first = aligned.iloc[0]
    invalid = first.index[first.isna() | (first <= 0)]
    if len(invalid):
        names = ", ".join(str(name) for name in invalid)
        raise ValueError(
            "Cannot start the basket. The first close is missing or not positive"
            f" for {names}."
        )

    shares = (basket.start_value / len(basket.tickers)) / first
    dollars_by_ticker = aligned.mul(shares, axis=1)
    history = aligned.div(first, axis=1).mul(100.0)
    history.insert(0, "dollars", dollars_by_ticker.sum(axis=1))
    for ticker in basket.tickers:
        history[f"{ticker}_dollars"] = dollars_by_ticker[ticker]
    history.index.name = "date"
    return history.reset_index()


PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _apply_chart_revision(fig: go.Figure, uirevision: str) -> go.Figure:
    fig.update_layout(uirevision=uirevision)
    return fig


def basket_figure(
    history: pd.DataFrame, basket: Basket, *, uirevision: str = "basket-0"
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["dollars"],
            name="Dollar value",
            line={"width": 3, "color": "#111827"},
        ),
        secondary_y=False,
    )
    for ticker in basket.tickers:
        fig.add_trace(
            go.Scatter(x=history["date"], y=history[ticker], name=ticker),
            secondary_y=True,
        )
    fig.update_layout(
        title={
            "text": f"{basket.label}, start ${basket.start_value:,.0f}",
            "font": {"size": 16},
        },
        autosize=True,
        margin={"l": 4, "r": 4, "t": 56, "b": 88},
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
            "itemsizing": "constant",
            "tracegroupgap": 4,
        },
    )
    fig.update_xaxes(title_text="Date", automargin=True, fixedrange=False)
    fig.update_yaxes(
        title_text="Dollars",
        tickprefix="$",
        separatethousands=True,
        automargin=True,
        fixedrange=False,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Indexed to 100",
        title_font_size=11,
        showgrid=False,
        automargin=True,
        fixedrange=True,
        secondary_y=True,
    )
    return _apply_chart_revision(fig, uirevision)


def holding_dollar_figure(
    history: pd.DataFrame,
    basket: Basket,
    ticker: str,
    *,
    uirevision: str = "holding-0",
) -> go.Figure:
    col = f"{ticker}_dollars"
    if col not in history.columns:
        raise ValueError(f"No dollar series for {ticker}.")

    sleeve = float(history[col].iloc[0])
    latest = float(history[col].iloc[-1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["dollars"],
            name="Basket dollars",
            line={"width": 3, "color": "#111827"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[col],
            name=f"{ticker} sleeve",
            line={"width": 2, "color": "#2563eb", "dash": "dash"},
        )
    )
    fig.update_layout(
        title={
            "text": f"{ticker} sleeve vs basket, equal-weight shares",
            "font": {"size": 16},
        },
        autosize=True,
        margin={"l": 4, "r": 4, "t": 56, "b": 72},
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
    )
    fig.update_xaxes(title_text="Date", automargin=True, fixedrange=False)
    fig.update_yaxes(
        title_text="Dollars",
        tickprefix="$",
        separatethousands=True,
        automargin=True,
        fixedrange=False,
    )
    fig.add_annotation(
        text=f"Start ${sleeve:,.0f} · Latest ${latest:,.2f}",
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        font={"size": 11, "color": "#4b5563"},
    )
    return _apply_chart_revision(fig, uirevision)


def main() -> None:
    st.set_page_config(page_title="Stock Lenses", layout="wide")
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.title("Stock Lenses")
    st.caption(
        "Each basket buys equal dollars of its holdings and keeps those share counts. "
        "Prices are cached in DuckDB, and missing or stale bars come from yfinance."
    )

    basket_keys = list(BASKETS.keys())
    chosen_key = st.selectbox(
        "Basket",
        options=basket_keys,
        format_func=lambda key: BASKETS[key].label,
    )
    basket = BASKETS[chosen_key]
    st.caption(f"Holdings: {', '.join(basket.tickers)}")

    try:
        with get_connection() as conn:
            with st.spinner(f"Loading {basket.label}..."):
                prices_by_ticker: dict[str, pd.DataFrame] = {}
                sources: dict[str, str] = {}
                for ticker in basket.tickers:
                    prices, source = get_prices(conn, ticker)
                    prices_by_ticker[ticker] = prices
                    sources[ticker] = source
        history = basket_history(prices_by_ticker, basket)
    except (ValueError, KeyError, OSError, duckdb.Error) as exc:
        st.error(str(exc))
        return

    fetched = [ticker for ticker in basket.tickers if sources[ticker] != "cache"]
    if not fetched:
        source_text = "Every ticker came from the DuckDB cache."
    elif len(fetched) == len(basket.tickers):
        source_text = "Every ticker was fetched from yfinance and stored in DuckDB."
    else:
        source_text = (
            "Fetched "
            + ", ".join(fetched)
            + " from yfinance and stored them in DuckDB. "
            "The other tickers came from the cache."
        )

    start_day = pd.Timestamp(history["date"].iloc[0]).date()
    end_day = pd.Timestamp(history["date"].iloc[-1]).date()
    start_value = float(history["dollars"].iloc[0])
    latest_value = float(history["dollars"].iloc[-1])
    change_pct = (latest_value / start_value - 1.0) * 100.0
    start_col, latest_col, days_col = st.columns(3)
    start_col.metric("Start", f"${start_value:,.0f}")
    latest_col.metric("Latest", f"${latest_value:,.2f}", delta=f"{change_pct:.1f}%")
    days_col.metric("Common days", f"{len(history)}")
    st.success(
        f"{basket.label} from {start_day} to {end_day}. {source_text}"
    )
    if "chart_revision" not in st.session_state:
        st.session_state.chart_revision = 0

    chart_mode = st.radio(
        "Chart",
        options=("Basket overview", "Selected holding (dollars)"),
        horizontal=False,
    )
    selected_ticker = basket.tickers[0]
    if chart_mode == "Selected holding (dollars)":
        selected_ticker = st.selectbox("Holding", options=list(basket.tickers))
        shares = basket_shares(prices_by_ticker, basket)
        st.caption(
            f"Fixed shares for {selected_ticker}: {shares[selected_ticker]:,.4f}. "
            f"Dollar value is shares times daily close."
        )

    reset_col, _ = st.columns([1, 3])
    if reset_col.button("Reset chart view", use_container_width=True):
        st.session_state.chart_revision += 1

    rev = f"{chosen_key}-{chart_mode}-{st.session_state.chart_revision}"
    st.caption(
        "The left axis is the basket in dollars. "
        "The right axis is each holding indexed to 100. "
        "Share counts stay fixed after the first common date. "
        "Pinch zoom is off so mobile panes do not trap a bad dual-axis range. "
        "Use Reset chart view to snap axes back."
        if chart_mode == "Basket overview"
        else "This path reprices one basket sleeve in dollars beside the full "
        "basket total, using the same equal-weight share count."
    )
    if chart_mode == "Basket overview":
        fig = basket_figure(history, basket, uirevision=rev)
    else:
        fig = holding_dollar_figure(
            history, basket, selected_ticker, uirevision=rev
        )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=f"plot-{rev}",
    )
    with st.expander("Basket series"):
        st.dataframe(history, use_container_width=True)


if __name__ == "__main__":
    main()
