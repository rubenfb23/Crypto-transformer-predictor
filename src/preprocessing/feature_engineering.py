import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_features(merged_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normaliza características numéricas y conserva timestamp."""
    df_clean = merged_dataframe.dropna().reset_index(drop=True)
    feature_columns = [col for col in df_clean.columns if col != "timestamp"]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_clean[feature_columns].values)
    df_scaled = pd.DataFrame(scaled_values, columns=feature_columns)
    df_scaled["timestamp"] = df_clean["timestamp"].values
    return df_scaled
