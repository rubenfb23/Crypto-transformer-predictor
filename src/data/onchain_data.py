import pandas as pd


class OnChainMetricsFetcher:
    """Extracts on-chain metrics for Bitcoin."""

    def __init__(self, node_client):
        self.node_client = node_client

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Devuelve DataFrame de métricas on-chain cada hora entre start y end."""
        # Generar índice horario entre start y end
        time_index = pd.date_range(start=start, end=end, freq="1H")
        df_metrics = pd.DataFrame(
            {
                "timestamp": time_index,
                "transaction_volume": 0.0,
                "active_addresses": 0,
                "mining_difficulty": 0.0,
                "hash_rate": 0.0,
            }
        )
        # TODO: reemplazar valores por lecturas reales desde node_client
        return df_metrics
