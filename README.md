# Stock Lenses

Minimal Streamlit app for exploring daily stock prices. Tickers are cached in DuckDB on a Docker volume; missing or stale data is fetched with yfinance.

## Run

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Stop with `Ctrl+C`, or in another terminal:

```bash
docker compose down
```

## How it works

1. Enter a ticker (e.g. `AAPL`) and click **Load**.
2. The app checks `/data/stocks.duckdb` for prices whose latest bar is within the last 3 days.
3. If found, it charts the cached series; otherwise it downloads ~1 year of daily bars via yfinance, stores them, then charts.

Data lives in the named Docker volume `stock-data` and persists across rebuilds.
