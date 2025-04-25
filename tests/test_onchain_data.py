import pandas as pd
import pytest
from unittest.mock import Mock

from src.data.onchain_data import OnChainMetricsFetcher


def test_onchain_fetcher_uses_client_and_formats_data():
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 02:00:00")
    mock_client = Mock()
    mock_data = {
        pd.Timestamp("2025-01-01 00:00:00"): {
            "transaction_volume": 100.5,
            "active_addresses": 5000,
            "mining_difficulty": 1.2e13,
            "hash_rate": 150e18,
        },
        pd.Timestamp("2025-01-01 01:00:00"): {
            "transaction_volume": 110.2,
            "active_addresses": 5100,
            "mining_difficulty": 1.21e13,
            "hash_rate": 152e18,
        },
        pd.Timestamp("2025-01-01 02:00:00"): {
            "transaction_volume": 105.8,
            "active_addresses": 5050,
            "mining_difficulty": 1.22e13,
            "hash_rate": 151e18,
        },
    }
    mock_client.get_metrics_for_timestamp.side_effect = lambda ts: mock_data.get(ts, {})

    fetcher = OnChainMetricsFetcher(node_client=mock_client)
    df = fetcher.fetch(start, end)

    assert isinstance(df, pd.DataFrame)
    # Timestamp is now the index, not a column
    expected_columns = [
        # "timestamp", # Removed
        "transaction_volume",
        "active_addresses",
        "mining_difficulty",
        "hash_rate",
    ]
    assert list(df.columns) == expected_columns

    # Check number of rows
    expected_timestamps = pd.date_range(start=start, end=end, freq="h")  # Use 'h'
    assert len(df) == len(expected_timestamps)

    # Check index (timestamps)
    pd.testing.assert_index_equal(
        df.index, pd.Index(expected_timestamps, name="timestamp")  # Check against index
    )

    # Check that the client method was called for each timestamp
    assert mock_client.get_metrics_for_timestamp.call_count == len(expected_timestamps)
    mock_client.get_metrics_for_timestamp.assert_any_call(expected_timestamps[0])

    # Check the data values (example for the first row)
    first_row_expected = mock_data[start]
    first_row_actual = df.loc[start]  # Access by index
    assert (
        first_row_actual["transaction_volume"]
        == first_row_expected["transaction_volume"]
    )
    assert (
        first_row_actual["active_addresses"] == first_row_expected["active_addresses"]
    )
    assert (
        first_row_actual["mining_difficulty"] == first_row_expected["mining_difficulty"]
    )
    assert first_row_actual["hash_rate"] == first_row_expected["hash_rate"]

    # Check the data values for the last row
    last_row_expected = mock_data[end]
    last_row_actual = df.loc[end]  # Access by index
    assert (
        last_row_actual["transaction_volume"] == last_row_expected["transaction_volume"]
    )
    assert last_row_actual["active_addresses"] == last_row_expected["active_addresses"]
    assert (
        last_row_actual["mining_difficulty"] == last_row_expected["mining_difficulty"]
    )
    assert last_row_actual["hash_rate"] == last_row_expected["hash_rate"]


def test_onchain_fetcher_generates_correct_time_index():
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 05:00:00")
    mock_client = Mock()
    mock_client.get_metrics_for_timestamp.return_value = {}  # Return empty dict
    fetcher = OnChainMetricsFetcher(node_client=mock_client)
    df = fetcher.fetch(start, end)
    assert isinstance(df, pd.DataFrame)
    # Timestamp is now the index
    expected_columns = [
        # "timestamp", # Removed
        "transaction_volume",
        "active_addresses",
        "mining_difficulty",
        "hash_rate",
    ]
    assert list(df.columns) == expected_columns

    # Check number of rows
    expected_timestamps = pd.date_range(start=start, end=end, freq="h")  # Use 'h'
    assert len(df) == len(expected_timestamps)

    # Check index (timestamps)
    pd.testing.assert_index_equal(
        df.index, pd.Index(expected_timestamps, name="timestamp")  # Check against index
    )
