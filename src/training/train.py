from datetime import datetime, timedelta
from dataclasses import dataclass
import ccxt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.ohlcv_data import OHLCVFetcher
from src.data.onchain_data import OnChainMetricsFetcher
from src.data.sentiment_data import SocialSentimentFetcher
from src.preprocessing.data_sync import synchronize_time_series
from src.preprocessing.feature_engineering import generate_features
from src.preprocessing.dataset import CryptoPriceDataset
from src.models.transformer_model import MultiModalTransformer


@dataclass
class TrainingConfig:
    lookback_hours: int = 24
    train_days: int = 30
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    model_dim: int = 64
    transformer_layers: int = 2


def fetch_raw_data(start_time: datetime, end_time: datetime):
    ohlcv = OHLCVFetcher(ccxt.binance(), ["BTC/USDT"], "1h")
    onchain = OnChainMetricsFetcher(node_client=None)
    sentiment = SocialSentimentFetcher(twitter_client=None, reddit_client=None)
    return (
        ohlcv.fetch(start_time, end_time),
        onchain.fetch(start_time, end_time),
        sentiment.fetch(start_time, end_time),
    )


def prepare_dataloader(ohlcv_df, onchain_df, sentiment_df, config: TrainingConfig):
    merged = synchronize_time_series(ohlcv_df, onchain_df, sentiment_df)
    features = generate_features(merged)
    dataset = CryptoPriceDataset(features, config.lookback_hours)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True), len(dataset)


def build_model(config: TrainingConfig):
    dims = {"ohlcv": 5, "onchain": 4, "sentiment": 4}
    model = MultiModalTransformer(
        input_dimensions=dims,
        model_dimension=config.model_dim,
        layer_count=config.transformer_layers,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def train_model():
    print("Iniciando entrenamiento")
    config = TrainingConfig()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=config.train_days)
    ohlcv_df, onchain_df, sentiment_df = fetch_raw_data(start_time, end_time)
    dataloader, dataset_size = prepare_dataloader(
        ohlcv_df, onchain_df, sentiment_df, config
    )
    model, optimizer, criterion = build_model(config)
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for o_seq, on_seq, s_seq, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(o_seq, on_seq, s_seq).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
        avg = total_loss / dataset_size
        print(f"Epoch {epoch}/{config.epochs}, Loss={avg:.4f}")
    torch.save(model.state_dict(), "transformer_model.pt")
    print("Entrenamiento completado.")
