# API Reference

This reference documents the main modules, classes, and functions provided by the Crypto-transformer-predictor package.

## src.main

### `main()`
Entry point for the application. Parses CLI arguments (`--mode`) and dispatches to one of:
- `train_model()`
- `evaluate_model()`
- `Dashboard().start()`

## src.data

### `OHLCVFetcher`
- `__init__(exchange, symbols, timeframe)`
- `fetch(start_time, end_time) -> DataFrame`
  Retrieves historical OHLCV (open/high/low/close/volume) data.

### `OnChainMetricsFetcher`
- `__init__(node_client)`
- `fetch(start_time, end_time) -> DataFrame`
  Retrieves on-chain metrics from a blockchain node.

### `SocialSentimentFetcher`
- `__init__(twitter_client, reddit_client)`
- `fetch(start_time, end_time) -> DataFrame`
  Retrieves aggregated social sentiment metrics.

## src.preprocessing

### `synchronize_time_series(ohlcv_df, onchain_df, sentiment_df) -> DataFrame`
Aligns and merges multiple time-series DataFrames on a common timestamp index.

### `generate_features(df) -> DataFrame`
Computes technical indicators and feature columns based on merged DataFrame.

### `CryptoPriceDataset`
- `__init__(data_frame, lookback_hours)`
- `__len__()`
- `__getitem__(index) -> (ohlcv_seq, onchain_seq, sentiment_seq, target)`
  PyTorch Dataset that returns input sequences and target price.

## src.models

### `MultiModalTransformer`
- `__init__(input_dimensions, model_dimension, layer_count)`
- `forward(ohlcv_seq, onchain_seq, sentiment_seq) -> Tensor`
  Multi-modal transformer model combining three input streams.

## src.training

### `train_model()`
High-level function to execute a training loop:
1. Fetch raw data
2. Prepare DataLoader
3. Build model, optimizer, loss
4. Iterate epochs and batches
5. Save `transformer_model.pt`

### `evaluate_model()`
Loads a saved model checkpoint and evaluates performance on test split.

## src.visualization

### `Dashboard`
- `start()`
  Launches an interactive dashboard for visualizing model predictions and metrics.
