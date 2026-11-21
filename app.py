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


EXTRAS_STATE_PREFIX = "basket_extras_"


def extras_state_key(basket_key: str) -> str:
    return f"{EXTRAS_STATE_PREFIX}{basket_key}"


def parse_ticker_input(raw: str) -> list[str]:
    tickers: list[str] = []
    for piece in raw.replace(",", " ").split():
        cleaned = piece.strip().upper()
        if cleaned:
            tickers.append(cleaned)
    return tickers


def normalize_extra_tickers(
    starter: tuple[str, ...], extras: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    starter_set = {t.upper() for t in starter}
    seen: set[str] = set()
    merged: list[str] = []
    for ticker in extras:
        cleaned = ticker.strip().upper()
        if not cleaned or cleaned in starter_set or cleaned in seen:
            continue
        seen.add(cleaned)
        merged.append(cleaned)
    return tuple(merged)


def starter_slot_dollars(basket: Basket) -> float:
    if basket.start_value <= 0 or not basket.tickers:
        raise ValueError(
            f"{basket.label} needs a positive start value and at least one ticker."
        )
    return basket.start_value / len(basket.tickers)


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


def _aligned_equal_weight_book(
    prices_by_ticker: dict[str, pd.DataFrame], basket: Basket
) -> tuple[pd.DataFrame, pd.Series]:
    slot = starter_slot_dollars(basket)
    aligned = align_closes(prices_by_ticker, basket.tickers)
    first = aligned.iloc[0]
    invalid = first.index[first.isna() | (first <= 0)]
    if len(invalid):
        names = ", ".join(str(name) for name in invalid)
        raise ValueError(
            "Cannot start the basket. The first close is missing or not positive"
            f" for {names}."
        )
    shares = slot / first
    return aligned, shares


def basket_shares(
    prices_by_ticker: dict[str, pd.DataFrame], basket: Basket
) -> dict[str, float]:
    _, shares = _aligned_equal_weight_book(prices_by_ticker, basket)
    return {str(ticker): float(shares[ticker]) for ticker in basket.tickers}


def basket_history(
    prices_by_ticker: dict[str, pd.DataFrame], basket: Basket
) -> pd.DataFrame:
    aligned, shares = _aligned_equal_weight_book(prices_by_ticker, basket)
    dollars_by_ticker = aligned.mul(shares, axis=1)
    history = aligned.div(aligned.iloc[0], axis=1).mul(100.0)
    history.insert(0, "dollars", dollars_by_ticker.sum(axis=1))
    for ticker in basket.tickers:
        history[f"{ticker}_dollars"] = dollars_by_ticker[ticker]
    history.index.name = "date"
    return history.reset_index()


def parallel_sleeve_shares(
    prices_by_ticker: dict[str, pd.DataFrame],
    basket: Basket,
    extra_ticker: str,
) -> tuple[float, pd.Series]:
    if extra_ticker not in prices_by_ticker:
        raise ValueError(f"No prices for {extra_ticker}.")
    try:
        aligned = align_closes(
            prices_by_ticker, (*basket.tickers, extra_ticker)
        )
    except ValueError as exc:
        raise ValueError(
            f"{extra_ticker} has no closes on the basket's common trading days."
        ) from exc
    extra = aligned[extra_ticker]
    first_close = float(extra.iloc[0])
    if first_close <= 0:
        raise ValueError(
            f"Cannot price {extra_ticker}. The first aligned close is not positive."
        )
    shares = starter_slot_dollars(basket) / first_close
    return shares, extra


def parallel_sleeve_dollars(
    prices_by_ticker: dict[str, pd.DataFrame],
    basket: Basket,
    extra_ticker: str,
) -> pd.DataFrame:
    shares, closes = parallel_sleeve_shares(prices_by_ticker, basket, extra_ticker)
    dollars = closes * shares
    frame = dollars.rename("dollars").reset_index()
    frame.insert(1, "ticker", extra_ticker)
    frame["shares"] = shares
    return frame


def parallel_sleeves_dollars(
    prices_by_ticker: dict[str, pd.DataFrame],
    basket: Basket,
    extra_tickers: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    return {
        ticker: parallel_sleeve_dollars(prices_by_ticker, basket, ticker)
        for ticker in extra_tickers
    }


PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
LOCK_INDEXED_Y_ON_OVERVIEW = True


def basket_figure(
    history: pd.DataFrame, basket: Basket, *, chart_id: str = "basket-0"
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
        uirevision=chart_id,
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
        fixedrange=LOCK_INDEXED_Y_ON_OVERVIEW,
        secondary_y=True,
    )
    return fig


def basket_dollars_figure(
    history: pd.DataFrame, basket: Basket, *, chart_id: str = "basket-dollars-0"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["dollars"],
            name="Basket dollars",
            line={"width": 3, "color": "#111827"},
        )
    )
    for ticker in basket.tickers:
        col = f"{ticker}_dollars"
        if col not in history.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history[col],
                name=f"{ticker} sleeve",
                line={"width": 1.5},
            )
        )
    fig.update_layout(
        title={
            "text": f"{basket.label} starter sleeves, start ${basket.start_value:,.0f}",
            "font": {"size": 16},
        },
        uirevision=chart_id,
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
    )
    return fig


def basket_figure_with_extras(
    history: pd.DataFrame,
    basket: Basket,
    extra_sleeves: dict[str, pd.DataFrame],
    *,
    chart_id: str = "basket-extras-0",
) -> go.Figure:
    fig = basket_figure(history, basket, chart_id=chart_id)
    for ticker, sleeve in extra_sleeves.items():
        fig.add_trace(
            go.Scatter(
                x=sleeve["date"],
                y=sleeve["dollars"],
                name=f"{ticker} (added)",
                line={"width": 2, "dash": "dot"},
            ),
            secondary_y=False,
        )
    slots = starter_slot_dollars(basket)
    title = f"{basket.label}, start ${basket.start_value:,.0f}"
    if extra_sleeves:
        title += f" · adds at ${slots:,.0f} slot"
    fig.update_layout(title={"text": title, "font": {"size": 16}})
    return fig


def compare_extras_figure(
    history: pd.DataFrame,
    basket: Basket,
    extra_sleeves: dict[str, pd.DataFrame],
    *,
    chart_id: str = "compare-extras-0",
) -> go.Figure:
    slot = starter_slot_dollars(basket)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["dollars"],
            name="Full basket",
            line={"width": 3, "color": "#111827"},
        )
    )
    for ticker in basket.tickers:
        col = f"{ticker}_dollars"
        if col in history.columns:
            fig.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history[col],
                    name=f"{ticker} (starter)",
                    line={"width": 1, "dash": "dash"},
                    opacity=0.55,
                )
            )
    for ticker, sleeve in extra_sleeves.items():
        fig.add_trace(
            go.Scatter(
                x=sleeve["date"],
                y=sleeve["dollars"],
                name=f"{ticker} (added)",
                line={"width": 2},
            )
        )
    fig.update_layout(
        title={
            "text": (
                f"Compare adds at one ${slot:,.0f} starter slot, "
                f"basket start ${basket.start_value:,.0f}"
            ),
            "font": {"size": 16},
        },
        uirevision=chart_id,
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
    )
    return fig


def holding_dollar_figure(
    history: pd.DataFrame,
    basket: Basket,
    ticker: str,
    *,
    chart_id: str = "holding-0",
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
        uirevision=chart_id,
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
    return fig


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
    st.caption(f"Starter holdings: {', '.join(basket.tickers)}")

    state_key = extras_state_key(chosen_key)
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    extra_tickers = normalize_extra_tickers(
        basket.tickers, tuple(st.session_state[state_key])
    )
    st.session_state[state_key] = list(extra_tickers)

    with st.expander("Add symbols", expanded=bool(extra_tickers)):
        slot = starter_slot_dollars(basket)
        st.caption(
            "Added tickers get one starter slot in dollars "
            f"(${slot:,.0f} each). "
            "They are not folded into the starter equal-weight book."
        )
        with st.form(f"add_symbols_form_{chosen_key}", clear_on_submit=True):
            add_col, btn_col = st.columns([3, 1])
            with add_col:
                ticker_draft = st.text_input(
                    "Add ticker",
                    placeholder="NVDA or BTC-USD",
                )
            with btn_col:
                add_clicked = st.form_submit_button(
                    "Add", use_container_width=True
                )

            if add_clicked:
                incoming = parse_ticker_input(ticker_draft)
                if incoming:
                    merged = normalize_extra_tickers(
                        basket.tickers,
                        (*st.session_state[state_key], *incoming),
                    )
                    st.session_state[state_key] = list(merged)
                    st.rerun()
                else:
                    st.warning("Enter at least one ticker symbol.")

        if extra_tickers:
            st.markdown("**Added symbols**")
            for ticker in extra_tickers:
                row_col, remove_col = st.columns([4, 1])
                row_col.markdown(f"**{ticker}**")
                if remove_col.button(
                    "Remove",
                    key=f"remove_{chosen_key}_{ticker}",
                    use_container_width=True,
                ):
                    st.session_state[state_key] = [
                        t for t in st.session_state[state_key] if t != ticker
                    ]
                    st.rerun()

    extra_tickers = normalize_extra_tickers(
        basket.tickers, tuple(st.session_state[state_key])
    )
    tickers_to_load = tuple(dict.fromkeys((*basket.tickers, *extra_tickers)))

    try:
        with get_connection() as conn:
            with st.spinner(f"Loading {basket.label}..."):
                prices_by_ticker: dict[str, pd.DataFrame] = {}
                sources: dict[str, str] = {}
                for ticker in tickers_to_load:
                    prices, source = get_prices(conn, ticker)
                    prices_by_ticker[ticker] = prices
                    sources[ticker] = source
        history = basket_history(prices_by_ticker, basket)
        extra_sleeves = parallel_sleeves_dollars(
            prices_by_ticker, basket, extra_tickers
        )
    except (ValueError, KeyError, OSError, duckdb.Error) as exc:
        st.error(str(exc))
        return

    fetched = [ticker for ticker in tickers_to_load if sources[ticker] != "cache"]
    if not fetched:
        source_text = "Every ticker came from the DuckDB cache."
    elif len(fetched) == len(tickers_to_load):
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

    chart_options = [
        "Basket overview",
        "Starter sleeves (dollars)",
        "Selected holding (dollars)",
    ]
    if extra_tickers:
        chart_options.append("Compare adds (dollars)")
    chart_mode = st.radio(
        "Chart",
        options=chart_options,
        horizontal=False,
    )
    reset_col, _ = st.columns([1, 3])
    if reset_col.button("Reset chart view", use_container_width=True):
        st.session_state.chart_revision += 1

    chart_id = f"{chosen_key}-{chart_mode}-{st.session_state.chart_revision}"
    slot = starter_slot_dollars(basket)
    if chart_mode == "Basket overview":
        st.caption(
            "Dollars on the left, starter holdings indexed to 100 on the right. "
            "Added tickers overlay as dollar sleeves sized to one starter slot "
            f"(${slot:,.0f}). Use Reset chart view if the axes look wrong."
        )
        fig = (
            basket_figure_with_extras(history, basket, extra_sleeves, chart_id=chart_id)
            if extra_sleeves
            else basket_figure(history, basket, chart_id=chart_id)
        )
    elif chart_mode == "Starter sleeves (dollars)":
        st.caption(
            "Each starter member repriced with fixed equal-weight shares. "
            "All sleeves start at the same dollar slot and sum to the basket."
        )
        fig = basket_dollars_figure(history, basket, chart_id=chart_id)
    elif chart_mode == "Compare adds (dollars)":
        st.caption(
            f"Added tickers vs the basket and each starter sleeve. "
            f"Each add uses one ${slot:,.0f} slot (start_value / "
            f"{len(basket.tickers)}), not a share of the full book."
        )
        fig = compare_extras_figure(
            history, basket, extra_sleeves, chart_id=chart_id
        )
    else:
        holding_options = list(basket.tickers) + list(extra_tickers)
        selected_ticker = st.selectbox("Holding", options=holding_options)
        if selected_ticker in basket.tickers:
            shares = basket_shares(prices_by_ticker, basket)
            st.caption(
                f"Fixed shares for {selected_ticker}: {shares[selected_ticker]:,.4f}. "
                f"Dollar value is shares times daily close."
            )
            fig = holding_dollar_figure(
                history, basket, selected_ticker, chart_id=chart_id
            )
        else:
            sleeve = extra_sleeves[selected_ticker]
            shares = float(sleeve["shares"].iloc[0])
            st.caption(
                f"Parallel slot for {selected_ticker}: {shares:,.4f} shares "
                f"from ${slot:,.0f} on the basket's first common day. "
                "Not included in basket dollars."
            )
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
                    x=sleeve["date"],
                    y=sleeve["dollars"],
                    name=f"{selected_ticker} add",
                    line={"width": 2, "color": "#2563eb", "dash": "dash"},
                )
            )
            fig.update_layout(
                title={
                    "text": f"{selected_ticker} add vs basket, one starter slot",
                    "font": {"size": 16},
                },
                uirevision=chart_id,
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
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=chart_id,
    )
    with st.expander("Basket series"):
        st.dataframe(history, use_container_width=True)


if __name__ == "__main__":
    main()
