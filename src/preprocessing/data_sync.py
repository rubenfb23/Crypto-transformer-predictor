import pandas as pd


def synchronize_time_series(
    *dataframes: pd.DataFrame, on: str = "timestamp"
) -> pd.DataFrame:
    """Une y alinea múltiples DataFrames en un índice común por la columna 'on'."""
    if not dataframes:
        raise ValueError("Se requieren al menos un DataFrame para sincronizar")
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on=on, how="inner")
    return merged
