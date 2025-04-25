import pandas as pd
import logging  # Import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OnChainMetricsFetcher:
    """Extracts on-chain metrics for Bitcoin."""

    def __init__(self, node_client):
        """
        Initializes the fetcher with a node client.

        Args:
            node_client: An object capable of fetching on-chain data.
                         It must have a method `get_metrics_for_timestamp(timestamp)`
                         which returns a dict with keys: 'transaction_volume',
                         'active_addresses', 'mining_difficulty', 'hash_rate'.
        """
        if node_client is None:
            # Log a warning if no client is provided, maybe raise error later
            logging.warning(
                "OnChainMetricsFetcher initialized without a node client. Fetch will return empty data."
            )
        self.node_client = node_client

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches hourly on-chain metrics between start and end timestamps.

        Args:
            start: The start timestamp (inclusive).
            end: The end timestamp (inclusive).

        Returns:
            A pandas DataFrame with hourly on-chain metrics.
            Columns: timestamp, transaction_volume, active_addresses,
                     mining_difficulty, hash_rate.
        """
        if self.node_client is None:
            logging.error("Cannot fetch data: node_client is not configured.")
            # Return an empty DataFrame matching the expected structure but with no rows
            # Keep original columns for consistency downstream
            return pd.DataFrame(
                {
                    # "timestamp": pd.Series(dtype='datetime64[ns]'), # No longer needed as column
                    "transaction_volume": pd.Series(dtype="float64"),
                    "active_addresses": pd.Series(dtype="int64"),
                    "mining_difficulty": pd.Series(dtype="float64"),
                    "hash_rate": pd.Series(dtype="float64"),
                }
            ).set_index(
                pd.Index([], dtype="datetime64[ns]", name="timestamp")
            )  # Set empty datetime index

        time_index = pd.date_range(
            start=start, end=end, freq="h", name="timestamp"
        )  # Use 'h' instead of '1H'
        metrics_data = []

        for timestamp in time_index:
            try:
                # Assume the client has this method, matching the test mock
                data = self.node_client.get_metrics_for_timestamp(timestamp)
                # Ensure all expected keys are present, provide defaults if not
                metrics_data.append(
                    {
                        "timestamp": timestamp,
                        "transaction_volume": data.get("transaction_volume", 0.0),
                        "active_addresses": data.get("active_addresses", 0),
                        "mining_difficulty": data.get("mining_difficulty", 0.0),
                        "hash_rate": data.get("hash_rate", 0.0),
                    }
                )
            except Exception as e:
                logging.error(
                    f"Failed to fetch or process on-chain data for {timestamp}: {e}"
                )
                # Append data with default/null values if fetch fails for a specific timestamp
                metrics_data.append(
                    {
                        "timestamp": timestamp,
                        "transaction_volume": pd.NA,  # Use pandas NA for missing numeric data
                        "active_addresses": pd.NA,
                        "mining_difficulty": pd.NA,
                        "hash_rate": pd.NA,
                    }
                )

        if not metrics_data:  # Handle case where loop doesn't run or all fetches fail
            return pd.DataFrame(
                {
                    # "timestamp": pd.Series(dtype='datetime64[ns]'), # No longer needed as column
                    "transaction_volume": pd.Series(dtype="float64"),
                    "active_addresses": pd.Series(dtype="int64"),  # Keep original types
                    "mining_difficulty": pd.Series(dtype="float64"),
                    "hash_rate": pd.Series(dtype="float64"),
                }
            ).set_index(
                pd.Index([], dtype="datetime64[ns]", name="timestamp")
            )  # Set empty datetime index

        # Create DataFrame from collected data
        df_metrics = pd.DataFrame(metrics_data)
        # Convert columns to appropriate types, handling potential pd.NA
        df_metrics["transaction_volume"] = pd.to_numeric(
            df_metrics["transaction_volume"], errors="coerce"
        )
        df_metrics["active_addresses"] = pd.to_numeric(
            df_metrics["active_addresses"], errors="coerce"
        ).astype(
            "Int64"
        )  # Use nullable integer
        df_metrics["mining_difficulty"] = pd.to_numeric(
            df_metrics["mining_difficulty"], errors="coerce"
        )
        df_metrics["hash_rate"] = pd.to_numeric(
            df_metrics["hash_rate"], errors="coerce"
        )

        # Set timestamp as index AFTER potential type conversions if needed
        df_metrics.set_index("timestamp", inplace=True)

        # Optional: Fill missing values if needed, e.g., forward fill
        # df_metrics.ffill(inplace=True)

        return df_metrics
