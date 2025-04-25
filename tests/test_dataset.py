import pandas as pd
import numpy as np
import torch

from src.preprocessing.dataset import CryptoPriceDataset


def test_crypto_price_dataset_basic():
    # Crear DataFrame de 6 puntos horarios
    timestamps = pd.date_range("2025-01-01", periods=6, freq="H")
    data = {
        "timestamp": timestamps,
        "open": np.arange(6),
        "high": np.arange(10, 16),
        "low": np.arange(20, 26),
        "close": np.arange(30, 36),
        "volume": np.arange(40, 46),
        "transaction_volume": np.arange(50, 56),
        "active_addresses": np.arange(60, 66),
        "mining_difficulty": np.arange(70, 76),
        "hash_rate": np.arange(80, 86),
        "tweet_volume": np.arange(90, 96),
        "retweet_count": np.arange(100, 106),
        "like_count": np.arange(110, 116),
        "sentiment_score": np.linspace(0.1, 0.6, 6),
    }
    df = pd.DataFrame(data)
    window_size = 3
    dataset = CryptoPriceDataset(df, window_size)

    # length should be len(df) - window_size
    assert len(dataset) == 6 - window_size

    # Test first item
    o, oc, s, target = dataset[0]
    # each should be torch.Tensor
    assert isinstance(o, torch.Tensor)
    assert isinstance(oc, torch.Tensor)
    assert isinstance(s, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    # Shapes: window_size x feature_count per modality
    assert o.shape == (window_size, 5)
    assert oc.shape == (window_size, 4)
    assert s.shape == (window_size, 4)

    # Target equals close at index window_size
    expected_target = df["close"].iloc[window_size]
    assert target.item() == expected_target

    # Test numeric values in first sequence for ohlcv: open col starts from 0
    np.testing.assert_array_equal(o[:, 0].numpy(), np.array([0, 1, 2]))
    # Test sentiment score in sentiment tensor
    np.testing.assert_allclose(
        s[:, 3].numpy(), df["sentiment_score"].values[:window_size]
    )
