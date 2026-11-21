import unittest
from datetime import date

import pandas as pd

from app import (
    BASKETS,
    Basket,
    basket_history,
    basket_shares,
    normalize_extra_tickers,
    parallel_sleeve_dollars,
    parallel_sleeve_shares,
    parse_ticker_input,
    starter_slot_dollars,
)


def prices(rows: list[tuple[object, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "close"])


class BasketRegistryTest(unittest.TestCase):
    def test_starter_baskets(self) -> None:
        self.assertEqual(set(BASKETS), {"stocks", "crypto", "metals"})
        stocks = BASKETS["stocks"]
        crypto = BASKETS["crypto"]
        metals = BASKETS["metals"]
        self.assertEqual(stocks.tickers, ("AAPL", "MSFT", "GOOGL", "AMZN"))
        self.assertEqual(crypto.tickers, ("BTC-USD", "ETH-USD", "SOL-USD"))
        self.assertEqual(metals.tickers, ("GC=F", "SI=F", "PL=F"))
        self.assertEqual(stocks.start_value, 10_000.0)
        self.assertEqual(crypto.start_value, 10_000.0)
        self.assertEqual(metals.start_value, 10_000.0)


class ExtraTickerHelpersTest(unittest.TestCase):
    def test_parse_ticker_input_splits_and_uppercases(self) -> None:
        self.assertEqual(parse_ticker_input(" nvda, tsla "), ["NVDA", "TSLA"])

    def test_normalize_extra_tickers_dedupes_and_skips_starter(self) -> None:
        starter = ("AAA", "BBB")
        extras = ("aaa", "CCC", "CCC", "BBB", "ddd")
        self.assertEqual(normalize_extra_tickers(starter, extras), ("CCC", "DDD"))

    def test_starter_slot_dollars(self) -> None:
        basket = Basket("t", "Test", ("A", "B", "C", "D"), 10_000.0)
        self.assertAlmostEqual(starter_slot_dollars(basket), 2_500.0)


class BasketHistoryTest(unittest.TestCase):
    def test_starts_at_start_value_and_holds_shares(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices(
                [
                    (date(2024, 1, 1), 10.0),
                    (date(2024, 1, 2), 12.0),
                    (date(2024, 1, 3), 20.0),
                ]
            ),
            "BBB": prices(
                [
                    (date(2024, 1, 1), 50.0),
                    (date(2024, 1, 2), 50.0),
                    (date(2024, 1, 3), 40.0),
                ]
            ),
        }
        history = basket_history(frames, basket)
        self.assertAlmostEqual(history["dollars"].iloc[0], 10_000.0)
        self.assertAlmostEqual(history["dollars"].iloc[1], 11_000.0)
        self.assertAlmostEqual(history["dollars"].iloc[2], 14_000.0)
        self.assertTrue((history.iloc[0][["AAA", "BBB"]] == 100.0).all())
        self.assertAlmostEqual(history["AAA"].iloc[1], 120.0)
        self.assertAlmostEqual(history["BBB"].iloc[1], 100.0)
        self.assertAlmostEqual(history["AAA"].iloc[2], 200.0)
        self.assertAlmostEqual(history["BBB"].iloc[2], 80.0)

    def test_mixed_duckdb_and_yfinance_date_types(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        duckdb_like = prices(
            [
                (pd.Timestamp("2024-01-01"), 10.0),
                (pd.Timestamp("2024-01-02"), 12.0),
                (pd.Timestamp("2024-01-03"), 20.0),
            ]
        )
        yfinance_like = prices(
            [
                (date(2024, 1, 1), 50.0),
                (date(2024, 1, 2), 50.0),
                (date(2024, 1, 3), 40.0),
            ]
        )
        history = basket_history({"AAA": duckdb_like, "BBB": yfinance_like}, basket)
        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(history["dollars"].iloc[0], 10_000.0)
        self.assertAlmostEqual(history["dollars"].iloc[1], 11_000.0)
        self.assertAlmostEqual(history["dollars"].iloc[2], 14_000.0)

    def test_index_components_start_at_100(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices([(date(2024, 1, 1), 10.0), (date(2024, 1, 2), 15.0)]),
            "BBB": prices([(date(2024, 1, 1), 40.0), (date(2024, 1, 2), 20.0)]),
        }
        history = basket_history(frames, basket)
        self.assertAlmostEqual(history["AAA"].iloc[0], 100.0)
        self.assertAlmostEqual(history["BBB"].iloc[0], 100.0)
        self.assertAlmostEqual(history["AAA"].iloc[1], 150.0)
        self.assertAlmostEqual(history["BBB"].iloc[1], 50.0)

    def test_no_overlap_raises(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices([(date(2024, 1, 1), 10.0)]),
            "BBB": prices([(date(2024, 1, 2), 20.0)]),
        }
        with self.assertRaises(ValueError):
            basket_history(frames, basket)

    def test_per_ticker_dollar_columns_match_fixed_shares(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices(
                [
                    (date(2024, 1, 1), 10.0),
                    (date(2024, 1, 2), 12.0),
                    (date(2024, 1, 3), 20.0),
                ]
            ),
            "BBB": prices(
                [
                    (date(2024, 1, 1), 50.0),
                    (date(2024, 1, 2), 50.0),
                    (date(2024, 1, 3), 40.0),
                ]
            ),
        }
        history = basket_history(frames, basket)
        shares = basket_shares(frames, basket)
        self.assertAlmostEqual(shares["AAA"], 500.0)
        self.assertAlmostEqual(shares["BBB"], 100.0)
        self.assertAlmostEqual(history["AAA_dollars"].iloc[0], 5_000.0)
        self.assertAlmostEqual(history["BBB_dollars"].iloc[0], 5_000.0)
        self.assertAlmostEqual(history["AAA_dollars"].iloc[1], 6_000.0)
        self.assertAlmostEqual(history["BBB_dollars"].iloc[1], 5_000.0)
        self.assertAlmostEqual(history["AAA_dollars"].iloc[2], 10_000.0)
        self.assertAlmostEqual(history["BBB_dollars"].iloc[2], 4_000.0)
        self.assertTrue(
            (
                history["AAA_dollars"] + history["BBB_dollars"] - history["dollars"]
            ).abs().max()
            < 1e-9
        )


class ParallelSleeveTest(unittest.TestCase):
    def test_parallel_sleeve_starts_at_one_starter_slot(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices(
                [
                    (date(2024, 1, 1), 10.0),
                    (date(2024, 1, 2), 12.0),
                ]
            ),
            "BBB": prices(
                [
                    (date(2024, 1, 1), 50.0),
                    (date(2024, 1, 2), 50.0),
                ]
            ),
            "CCC": prices(
                [
                    (date(2024, 1, 1), 25.0),
                    (date(2024, 1, 2), 30.0),
                ]
            ),
        }
        history = basket_history(frames, basket)
        sleeve = parallel_sleeve_dollars(frames, basket, "CCC")
        slot = starter_slot_dollars(basket)
        self.assertAlmostEqual(sleeve["dollars"].iloc[0], slot)
        self.assertAlmostEqual(sleeve["dollars"].iloc[1], slot * 30.0 / 25.0)
        self.assertEqual(len(sleeve), len(history))

    def test_added_ticker_does_not_change_basket_book(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        starter_only = {
            "AAA": prices([(date(2024, 1, 1), 10.0), (date(2024, 1, 2), 12.0)]),
            "BBB": prices([(date(2024, 1, 1), 50.0), (date(2024, 1, 2), 50.0)]),
        }
        with_extra = {
            **starter_only,
            "CCC": prices([(date(2024, 1, 1), 25.0), (date(2024, 1, 2), 30.0)]),
        }
        base = basket_history(starter_only, basket)
        unchanged = basket_history(with_extra, basket)
        pd.testing.assert_frame_equal(
            base[["date", "dollars", "AAA", "BBB"]],
            unchanged[["date", "dollars", "AAA", "BBB"]],
        )

    def test_parallel_shares_use_slot_not_full_book(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices([(date(2024, 1, 1), 10.0)]),
            "BBB": prices([(date(2024, 1, 1), 50.0)]),
            "CCC": prices([(date(2024, 1, 1), 25.0)]),
        }
        shares, _ = parallel_sleeve_shares(frames, basket, "CCC")
        self.assertAlmostEqual(shares, 200.0)

    def test_extra_without_starter_overlap_raises(self) -> None:
        basket = Basket("t", "Test", ("AAA", "BBB"), 10_000.0)
        frames = {
            "AAA": prices([(date(2024, 1, 1), 10.0)]),
            "BBB": prices([(date(2024, 1, 1), 50.0)]),
            "CCC": prices([(date(2024, 1, 2), 25.0)]),
        }
        with self.assertRaises(ValueError):
            parallel_sleeve_dollars(frames, basket, "CCC")


if __name__ == "__main__":
    unittest.main()