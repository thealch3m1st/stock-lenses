# Stock Lenses

Streamlit app for three equal-weight baskets. Daily prices are cached in DuckDB on a Docker volume. The app fetches a ticker from yfinance when the cache is missing or the newest bar is older than 3 days.

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

## Tests

```bash
docker run --rm -v "$PWD":/workspace -w /workspace python:3.12-slim \
  sh -c "pip install -q -r requirements.txt && python -m unittest test_baskets -v"
```

## Switch baskets

1. Open the app.
2. Under **Basket**, choose **Dollar stocks**, **Crypto**, or **Precious metals**.
3. Read the chart.

**Dollar stocks** holds AAPL, MSFT, GOOGL, and AMZN. **Crypto** holds BTC-USD, ETH-USD, and SOL-USD. **Precious metals** holds COMEX futures GC=F (gold), SI=F (silver), and PL=F (platinum). Each basket starts at $10,000. On the first date present for every holding, the app buys an equal dollar amount of each and keeps those share counts.

The left axis is the basket's dollar value. The right axis is each holding indexed to 100 on that same first date.

## Add tickers

Open **Add symbols**, type a ticker (for example `NVDA` or `BTC-USD`), and tap **Add**. Added symbols are stored per basket in the session. They are not merged into the equal-weight starter book. Each add is repriced as a parallel dollar sleeve sized to one starter slot (`start_value / number of starter holdings`). Tap **Remove** to drop a symbol from the list.

Added symbols load through the same DuckDB cache and `get_prices` path as starter holdings. On **Basket overview**, added sleeves overlay the basket dollar line on the left axis. With at least one add, **Compare adds (dollars)** plots the full basket, every starter sleeve, and each add on one dollar chart.

## Chart modes

| Mode | What you see |
| --- | --- |
| **Basket overview** | Basket dollars plus indexed starter members; added symbols overlay as dotted dollar sleeves. |
| **Starter sleeves (dollars)** | Basket dollars and each starter member repriced in dollars (fixed equal-weight shares). |
| **Selected holding (dollars)** | One starter member or added symbol vs basket dollars. |
| **Compare adds (dollars)** | Shown when adds exist. Basket, starter sleeves, and adds on one dollar chart. |

On a phone, use the **Basket** dropdown instead of side-by-side chips. Metrics stack in a single column near 390px width, and the chart fits the screen width with a compact legend under the plot. Pinch zoom is disabled on the overview chart so a bad dual-axis zoom cannot stick. Tap **Reset chart view** to force axes back to the full range.

## Price cache

For each ticker, `get_prices` reads the `daily_prices` table in `/data/stocks.duckdb`. A series whose latest bar is within 3 days is used as stored. Otherwise the app downloads about 1 year of daily bars, replaces that ticker's rows, and builds the basket from those closes.