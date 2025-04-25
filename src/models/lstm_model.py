import torch.nn as nn


class LstmBenchmarkModel(nn.Module):
    """LSTM baseline model para predicción de precios."""

    def __init__(
        self, input_feature_count: int, hidden_dimension: int, layer_count: int
    ):
        super().__init__()
        self.recurrent_network = nn.LSTM(
            input_feature_count,
            hidden_dimension,
            num_layers=layer_count,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, feature_sequence):
        """Forward pass que retorna predicción de precio a partir de la secuencia."""
        outputs, _ = self.recurrent_network(feature_sequence)
        last_step = outputs[:, -1, :]
        return self.output_layer(last_step)
