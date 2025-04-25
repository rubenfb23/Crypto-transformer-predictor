import pandas as pd
import pytest
from src.data.ohlcv_data import OHLCVFetcher


class DummyClient:
    def __init__(self, data):
        self.data = data

    def fetch_ohlcv(self, symbol, interval, since=None, limit=None):
        # Ignora parámetros y retorna datos dummy
        return self.data


def test_ohlcv_fetcher_dataframe_columns_and_length():
    # Preparar datos dummy en ms
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 02:00:00")
    ms1 = int(start.timestamp() * 1000)
    ms2 = ms1 + 3600 * 1000
    ms3 = ms1 + 2 * 3600 * 1000
    dummy_records = [
        [ms1, 1.0, 2.0, 0.5, 1.5, 10.0],
        [ms2, 1.5, 2.5, 1.0, 2.0, 12.0],
        [ms3, 2.0, 3.0, 1.5, 2.5, 15.0],
    ]
    client = DummyClient(dummy_records)
    fetcher = OHLCVFetcher(client, ["BTC/USDT"], "1h")
    df = fetcher.fetch(start, end)

    # Verificar tipo y columnas
    assert isinstance(df, pd.DataFrame)
    expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_columns
    # Debe contener tantas filas como registros
    assert len(df) == len(dummy_records)

    # Verificar conversión de timestamp y filtrado
    expected_ts = pd.to_datetime([ms1, ms2, ms3], unit="ms")
    pd.testing.assert_series_equal(
        df["timestamp"], pd.Series(expected_ts), check_names=False
    )
