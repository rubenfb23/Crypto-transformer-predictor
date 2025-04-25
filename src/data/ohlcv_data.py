import pandas as pd
import ccxt


class OHLCVFetcher:
    def __init__(self, api_client, symbols, interval):
        self.api_client = api_client
        self.symbols = symbols
        self.interval = interval

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Retorna un DataFrame con OHLCV entre start y end para el primer símbolo."""
        all_records = []
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        symbol = self.symbols[0]
        # paginar hasta cubrir el rango
        while since_ms < end_ms:
            batch = self.api_client.fetch_ohlcv(
                symbol, self.interval, since=since_ms, limit=1000
            )
            if not batch:
                break
            all_records.extend(batch)
            last_timestamp = batch[-1][0]
            # avanzar al siguiente punto (evita registros duplicados)
            since_ms = last_timestamp + 1
        # convertir a DataFrame
        df = pd.DataFrame(
            all_records, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # filtrar por rango de fechas
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        return df.loc[mask].reset_index(drop=True)
