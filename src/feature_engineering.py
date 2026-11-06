"""Feature engineering module for EPred.

Generates temporal, historical lag/rolling, and indoor/outdoor environmental features
while strictly preventing future-data leakage.
"""

import logging
from typing import List, Optional
import pandas as pd
import numpy as np

from src.config import (
    INDOOR_TEMP_COLS,
    INDOOR_HUMIDITY_COLS,
    OUTDOOR_BUILDING_TEMP_COL,
    OUTDOOR_BUILDING_HUMIDITY_COL,
    FEATURE_ORDER,
    TIME_FEATURES,
    LAG_FEATURES,
    ENVIRONMENTAL_FEATURES
)

logger = logging.getLogger(__name__)


def generate_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Generates calendar and cyclical time features from datetime column.

    Features generated:
    - Hour, Minute, Day_of_Week, Is_Weekend, Month
    - Hour_Sin, Hour_Cos (cyclical 24-hour cycle)
    - Weekday_Sin, Weekday_Cos (cyclical 7-day cycle)
    """
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    dt = df[date_col].dt
    df["Hour"] = dt.hour
    df["Minute"] = dt.minute
    df["Day_of_Week"] = dt.day_of_week
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Month"] = dt.month

    # Continuous 24h fractional cycle
    continuous_hour = df["Hour"] + df["Minute"] / 60.0
    df["Hour_Sin"] = np.sin(2.0 * np.pi * continuous_hour / 24.0)
    df["Hour_Cos"] = np.cos(2.0 * np.pi * continuous_hour / 24.0)

    # Weekly cyclical encoding
    df["Weekday_Sin"] = np.sin(2.0 * np.pi * df["Day_of_Week"] / 7.0)
    df["Weekday_Cos"] = np.cos(2.0 * np.pi * df["Day_of_Week"] / 7.0)

    return df


def generate_lag_features(
    df: pd.DataFrame,
    target_col: str = "Appliances"
) -> pd.DataFrame:
    """Generates lag and rolling window features for energy consumption.

    Strictly uses current and past observations only (at or before time t).
    Never uses future observations (t+1, etc.).
    """
    df = df.copy()
    series = df[target_col]

    # Discrete lags
    df["Appliances_Lag_1"] = series.shift(1)
    df["Appliances_Lag_2"] = series.shift(2)
    df["Appliances_Lag_3"] = series.shift(3)
    df["Appliances_Lag_6"] = series.shift(6)
    df["Appliances_Lag_12"] = series.shift(12)

    # Rolling statistics: closed='right' means current observation is included,
    # which is available at or before prediction time t.
    df["Appliances_Rolling_Mean_3"] = series.rolling(window=3, min_periods=1).mean()
    df["Appliances_Rolling_Mean_6"] = series.rolling(window=6, min_periods=1).mean()
    df["Appliances_Rolling_Mean_12"] = series.rolling(window=12, min_periods=1).mean()
    df["Appliances_Rolling_Std_6"] = series.rolling(window=6, min_periods=1).std().fillna(0.0)

    return df


def generate_environmental_features(
    df: pd.DataFrame,
    indoor_temp_cols: Optional[List[str]] = None,
    indoor_hum_cols: Optional[List[str]] = None,
    outdoor_temp_col: str = OUTDOOR_BUILDING_TEMP_COL,
    outdoor_hum_col: str = OUTDOOR_BUILDING_HUMIDITY_COL
) -> pd.DataFrame:
    """Computes aggregated indoor temperature and humidity and indoor/outdoor gradients.

    Features generated:
    - Avg_Indoor_Temp: Mean of indoor temperature sensors (T1-T5, T7-T9)
    - Avg_Indoor_Humidity: Mean of indoor humidity sensors (RH_1-RH_5, RH_7-RH_9)
    - Temp_Difference: Indoor average minus outdoor building-side (T6)
    - Humidity_Difference: Indoor average minus outdoor building-side (RH_6)
    """
    df = df.copy()
    in_temp = indoor_temp_cols or INDOOR_TEMP_COLS
    in_hum = indoor_hum_cols or INDOOR_HUMIDITY_COLS

    df["Avg_Indoor_Temp"] = df[in_temp].mean(axis=1)
    df["Avg_Indoor_Humidity"] = df[in_hum].mean(axis=1)

    df["Temp_Difference"] = df["Avg_Indoor_Temp"] - df[outdoor_temp_col]
    df["Humidity_Difference"] = df["Avg_Indoor_Humidity"] - df[outdoor_hum_col]

    return df


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Executes full feature engineering pipeline in sequence."""
    df = generate_time_features(df)
    df = generate_lag_features(df)
    df = generate_environmental_features(df)
    return df
