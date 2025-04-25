import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
from dataclasses import dataclass
import ccxt
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.ohlcv_data import OHLCVFetcher
from src.data.onchain_data import OnChainMetricsFetcher
from src.data.sentiment_data import SocialSentimentFetcher
from src.preprocessing.data_sync import synchronize_time_series
from src.preprocessing.feature_engineering import generate_features
from src.preprocessing.dataset import CryptoPriceDataset
from src.models.transformer_model import MultiModalTransformer


@dataclass
class EvaluationConfig:
    lookback_hours: int = 24
    eval_days: int = 7
    batch_size: int = 32
    model_checkpoint: str = "transformer_model.pt"


def fetch_evaluation_data(config: EvaluationConfig):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=config.eval_days)
    ohlcv = OHLCVFetcher(ccxt.binance(), ["BTC/USDT"], "1h")
    onchain = OnChainMetricsFetcher(node_client=None)
    sentiment = SocialSentimentFetcher(twitter_client=None, reddit_client=None)
    return (
        ohlcv.fetch(start_time, end_time),
        onchain.fetch(start_time, end_time),
        sentiment.fetch(start_time, end_time),
    )


def prepare_evaluation_loader(
    ohlcv_df, onchain_df, sentiment_df, config: EvaluationConfig
):
    merged = synchronize_time_series(ohlcv_df, onchain_df, sentiment_df)
    features = generate_features(merged)
    dataset = CryptoPriceDataset(features, config.lookback_hours)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


def load_model(checkpoint_path: str):
    dims = {"ohlcv": 5, "onchain": 4, "sentiment": 4}
    model = MultiModalTransformer(
        input_dimensions=dims, model_dimension=64, layer_count=2
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def compute_and_print_metrics(preds: np.ndarray, targets: np.ndarray):
    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)
    mape = np.mean(np.abs((targets - preds) / targets)) * 100
    r2 = r2_score(targets, preds)
    directions_true = np.sign(targets[1:] - targets[:-1])
    directions_pred = np.sign(preds[1:] - targets[:-1])
    direction_acc = np.mean(directions_true == directions_pred) * 100
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")
    print(f"Direction accuracy: {direction_acc:.2f}%")


def evaluate_model():
    print("Iniciando evaluación")
    config = EvaluationConfig()
    ohlcv_df, onchain_df, sentiment_df = fetch_evaluation_data(config)
    loader = prepare_evaluation_loader(ohlcv_df, onchain_df, sentiment_df, config)
    model = load_model(config.model_checkpoint)
    preds, targets = [], []
    with torch.no_grad():
        for o_seq, on_seq, s_seq, t in loader:
            output = model(o_seq, on_seq, s_seq).squeeze()
            preds.append(output.numpy())
            targets.append(t.numpy())
    preds_arr = np.concatenate(preds)
    targets_arr = np.concatenate(targets)
    compute_and_print_metrics(preds_arr, targets_arr)
