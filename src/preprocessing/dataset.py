import torch
from torch.utils.data import Dataset
import pandas as pd


class CryptoPriceDataset(Dataset):
    """Dataset que genera secuencias de ventana deslizante y targets de precio siguiente."""

    def __init__(self, features_dataframe: pd.DataFrame, window_size: int):
        # Conversión a numpy excluyendo timestamp
        df = features_dataframe.copy().reset_index(drop=True)
        self.timestamps = df["timestamp"].values
        feature_df = df.drop(columns=["timestamp"])
        self.feature_columns = feature_df.columns.tolist()
        values = feature_df.values.astype(float)
        self.window_size = window_size
        sequences = []
        targets = []
        close_idx = self.feature_columns.index("close")
        for i in range(len(values) - window_size):
            sequences.append(values[i : i + window_size])
            targets.append(values[i + window_size, close_idx])
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # Separar secuencias por modalidad según columnas
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        onchain_cols = [
            "transaction_volume",
            "active_addresses",
            "mining_difficulty",
            "hash_rate",
        ]
        sentiment_cols = [
            "tweet_volume",
            "retweet_count",
            "like_count",
            "sentiment_score",
        ]
        col_index = {name: idx for idx, name in enumerate(self.feature_columns)}
        o_idx = [col_index[c] for c in ohlcv_cols]
        oc_idx = [col_index[c] for c in onchain_cols]
        s_idx = [col_index[c] for c in sentiment_cols]
        self.ohlcv_sequences = self.sequences[:, :, o_idx]
        self.onchain_sequences = self.sequences[:, :, oc_idx]
        self.sentiment_sequences = self.sequences[:, :, s_idx]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Retorna tupla de secuencias por modalidad y el target
        return (
            self.ohlcv_sequences[idx],
            self.onchain_sequences[idx],
            self.sentiment_sequences[idx],
            self.targets[idx],
        )
