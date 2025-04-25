import torch
from torch import nn


class MultiModalTransformer(nn.Module):
    """Transformer que consume secuencias de OHLCV, on-chain y sentimiento para regresión de precios."""

    def __init__(
        self, input_dimensions: dict[str, int], model_dimension: int, layer_count: int
    ):
        super().__init__()
        # capas de proyección (embeddings) por modalidad
        self.ohlcv_embedding = nn.Linear(input_dimensions["ohlcv"], model_dimension)
        self.onchain_embedding = nn.Linear(input_dimensions["onchain"], model_dimension)
        self.sentiment_embedding = nn.Linear(
            input_dimensions["sentiment"], model_dimension
        )
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=4,
            dim_feedforward=model_dimension * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layer_count
        )
        # cabeza de regresión
        self.regressor = nn.Linear(model_dimension, 1)

    def forward(self, ohlcv_sequence, onchain_sequence, sentiment_sequence):
        """Forward pass sobre la secuencia multimodal."""
        # proyectar cada modalidad al espacio de modelo
        e_ohlcv = self.ohlcv_embedding(ohlcv_sequence)
        e_onchain = self.onchain_embedding(onchain_sequence)
        e_sentiment = self.sentiment_embedding(sentiment_sequence)
        # fusionar modalidades por suma
        fused = e_ohlcv + e_onchain + e_sentiment
        # pasar por el encoder Transformer
        encoded = self.transformer_encoder(fused)
        # usar salida del último paso de tiempo para regresión
        last_hidden = encoded[:, -1, :]
        return self.regressor(last_hidden)
