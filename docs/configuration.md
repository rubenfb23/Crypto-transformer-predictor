# Configuration

This document describes available CLI modes and configurable parameters.

## Modes

- `train`: Trains the transformer model on historical data.
- `evaluate`: Evaluates a saved model against test data.
- `dashboard`: Launches an interactive dashboard for visual analysis.

## Training Parameters

The `TrainingConfig` dataclass defines default values. You can override these by modifying code or extending functionality.

| Parameter             | Type    | Default | Description                          |
|-----------------------|---------|---------|--------------------------------------|
| `lookback_hours`      | int     | 24      | Number of hours of history per sample |
| `train_days`          | int     | 30      | Total days of data to fetch          |
| `batch_size`          | int     | 32      | Samples per training batch           |
| `epochs`              | int     | 10      | Number of training epochs            |
| `learning_rate`       | float   | 1e-3    | Learning rate for optimizer          |
| `model_dim`           | int     | 64      | Dimension of transformer embeddings  |
| `transformer_layers`  | int     | 2       | Number of transformer encoder layers |

## File Paths and Output

- Trained model is saved as `transformer_model.pt` in the working directory.
- Logs and progress are printed to stdout by default.