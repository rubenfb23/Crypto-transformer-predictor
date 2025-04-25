import pandas as pd
import numpy as np
from src.preprocessing.feature_engineering import generate_features


def test_generate_features_preserves_timestamp_and_scales():
    timestamps = pd.date_range("2025-01-01", periods=4, freq="H")
    # Crear features a y b con valores lineales
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        }
    )
    result = generate_features(df)
    # Debe conservar timestamps
    assert "timestamp" in result.columns
    pd.testing.assert_series_equal(
        result["timestamp"], pd.Series(timestamps), check_names=False
    )
    # Verificar que la media de a y b sea aproximadamente 0 y varianza 1
    a_scaled = result["a"].values
    b_scaled = result["b"].values
    assert np.allclose(np.mean(a_scaled), 0.0, atol=1e-6)
    assert np.allclose(np.std(a_scaled), 1.0, atol=1e-6)
    assert np.allclose(np.mean(b_scaled), 0.0, atol=1e-6)
    assert np.allclose(np.std(b_scaled), 1.0, atol=1e-6)
