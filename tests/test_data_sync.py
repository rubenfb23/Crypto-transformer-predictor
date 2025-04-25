import pandas as pd
from src.preprocessing.data_sync import synchronize_time_series


def test_synchronize_time_series_inner_join_all():
    timestamps = pd.date_range("2025-01-01", periods=3, freq="H")
    df1 = pd.DataFrame({"timestamp": timestamps, "a": [1, 2, 3]})
    df2 = pd.DataFrame({"timestamp": timestamps, "b": [4, 5, 6]})
    merged = synchronize_time_series(df1, df2)
    assert list(merged.columns) == ["timestamp", "a", "b"]
    assert len(merged) == 3


def test_synchronize_time_series_inner_join_partial():
    timestamps_full = pd.date_range("2025-01-01", periods=3, freq="H")
    timestamps_partial = pd.date_range("2025-01-01", periods=2, freq="H")
    df1 = pd.DataFrame({"timestamp": timestamps_full, "a": [1, 2, 3]})
    df2 = pd.DataFrame({"timestamp": timestamps_partial, "b": [7, 8]})
    merged = synchronize_time_series(df1, df2)
    assert len(merged) == 2
    assert all(merged["timestamp"] == timestamps_partial)
