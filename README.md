# Stock Lenses

Streamlit app for two equal-weight baskets. Daily prices are cached in DuckDB on a Docker volume. The app fetches a ticker from yfinance when the cache is missing or the newest bar is older than 3 days.

## Run

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

Stop the app with `Ctrl+C`. To remove the container:

```bash
docker compose down
```

The `stock-data` volume keeps the DuckDB file across rebuilds.

## Switch baskets

1. Open the app.
2. Under **Basket**, choose **Dollar stocks** or **Crypto**.
3. Read the chart.

**Dollar stocks** holds AAPL, MSFT, GOOGL, and AMZN. **Crypto** holds BTC-USD, ETH-USD, and SOL-USD. Each basket starts at $10,000. On the first date present for every holding, the app buys an equal dollar amount of each and keeps those share counts.

The left axis is the basket's dollar value. The right axis is each holding indexed to 100 on that same first date.

## Price cache

For each ticker, `get_prices` reads the `daily_prices` table in `/data/stocks.duckdb`. A series whose latest bar is within 3 days is used as stored. Otherwise the app downloads about 1 year of daily bars, replaces that ticker's rows, and builds the basket from those closes.
