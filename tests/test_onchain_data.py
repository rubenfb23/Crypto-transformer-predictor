import pandas as pd
import pytest

from src.data.onchain_data import OnChainMetricsFetcher


def test_onchain_fetcher_generates_hourly_metrics():
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 05:00:00")
    fetcher = OnChainMetricsFetcher(node_client=None)
    df = fetcher.fetch(start, end)
    assert isinstance(df, pd.DataFrame)
    expected_columns = [
        "timestamp",
        "transaction_volume",
        "active_addresses",
        "mining_difficulty",
        "hash_rate",
    ]
    assert list(df.columns) == expected_columns
    # Chequear número de filas = horas + 1
    assert len(df) == len(pd.date_range(start=start, end=end, freq="1H"))
    # Chequear timestamps equiespaciados
    pd.testing.assert_series_equal(
        df["timestamp"],
        pd.Series(pd.date_range(start=start, end=end, freq="1H")),
        check_names=False,
    )
