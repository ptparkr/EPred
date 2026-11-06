"""Data preprocessing module for EPred.

Performs data cleaning, outlier diagnostics, target generation (single and multi-horizon),
feature preparation, chronological train/val/test splitting, and fitting preprocessing scalers on training data only.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    COLUMNS_TO_DROP,
    TARGET_COL,
    TARGET_COL_30MIN,
    TARGET_COL_60MIN,
    ALL_TARGET_COLS,
    BASE_TARGET_COL,
    FEATURE_ORDER,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.data_loader import DataLoader
from src.feature_engineering import generate_all_features

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for EPred."""

    def __init__(self, raw_data_path: Path = RAW_DATA_PATH):
        self.raw_data_path = Path(raw_data_path)
        self.data_loader = DataLoader(self.raw_data_path)
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns = FEATURE_ORDER

    def investigate_outliers(self, df: pd.DataFrame, col: str = BASE_TARGET_COL) -> Dict[str, Any]:
        """Analyzes distribution and outliers for the target column.

        Critically, high appliance consumption values reflect real-world episodic device
        usage (e.g. ovens, heaters, laundry machines) and must NOT be deleted blindly.
        """
        series = df[col]
        q25 = float(series.quantile(0.25))
        q50 = float(series.median())
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = max(0.0, q25 - 1.5 * iqr)
        upper_bound = q75 + 1.5 * iqr
        p99 = float(series.quantile(0.99))
        max_val = float(series.max())
        min_val = float(series.min())

        outliers_count = int((series > upper_bound).sum())
        outliers_pct = float(outliers_count / len(series) * 100)

        outlier_summary = {
            "feature": col,
            "min": min_val,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "iqr": iqr,
            "p99": p99,
            "max": max_val,
            "upper_bound_1_5_iqr": upper_bound,
            "high_outliers_count": outliers_count,
            "high_outliers_percentage": round(outliers_pct, 2),
            "treatment": "Retained legitimate high-consumption appliance operational cycles."
        }
        logger.info(f"Outlier diagnostic for '{col}': {outlier_summary}")
        return outlier_summary

    def prepare_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes full data loading, feature engineering, and multi-horizon target creation.

        Multi-horizon target generation:
            Appliances_Next_10min = Appliances.shift(-1)  (t + 10 min)
            Appliances_Next_30min = Appliances.shift(-3)  (t + 30 min)
            Appliances_Next_60min = Appliances.shift(-6)  (t + 60 min)
        Warmup rows (first 12) where Appliances_Lag_12 is NaN and terminal rows (final 6)
        where 60-min target is unavailable are trimmed cleanly.
        """
        df = self.data_loader.load_data()
        inspection_report = self.data_loader.inspect_data(df)
        outlier_report = self.investigate_outliers(df, BASE_TARGET_COL)

        # Drop non-predictive random variables
        cols_to_remove = [c for c in COLUMNS_TO_DROP if c in df.columns]
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)

        # Generate all features
        df = generate_all_features(df)

        # Generate multi-horizon forecasting targets
        df[TARGET_COL] = df[BASE_TARGET_COL].shift(-1)
        df[TARGET_COL_30MIN] = df[BASE_TARGET_COL].shift(-3)
        df[TARGET_COL_60MIN] = df[BASE_TARGET_COL].shift(-6)

        # Exclude rows where targets or features have NaNs
        initial_len = len(df)
        df = df.dropna(subset=ALL_TARGET_COLS + FEATURE_ORDER).reset_index(drop=True)
        dropped_rows = initial_len - len(df)
        logger.info(f"Dropped {dropped_rows} rows (12 lag warmup rows + 6 terminal target rows). Total ready: {len(df)}")

        metadata = {
            "inspection": inspection_report,
            "outliers": outlier_report,
            "final_rows": len(df),
            "feature_count": len(FEATURE_ORDER),
            "target_columns": ALL_TARGET_COLS,
            "primary_target": TARGET_COL,
        }

        return df, metadata

    def chronological_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits time-series chronologically into Train (70%), Val (15%), Test (15%).

        Guarantees that train < val < test in timestamp order.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        logger.info(
            f"Chronological split completed:\n"
            f"  Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})\n"
            f"  Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})\n"
            f"  Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})"
        )

        return train_df, val_df, test_df

    def fit_scaler(self, train_features: pd.DataFrame) -> StandardScaler:
        """Fits StandardScaler ONLY on training data to prevent data leakage."""
        self.scaler = StandardScaler()
        self.scaler.fit(train_features[self.feature_columns])
        return self.scaler

    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """Applies fitted scaler to features."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_scaler first.")
        return self.scaler.transform(features[self.feature_columns])

    def save_processed_datasets(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Saves processed and split datasets to disk without modifying raw data."""
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(PROCESSED_DATA_PATH, index=False)
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        val_df.to_csv(VAL_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info(f"Saved processed datasets to {PROCESSED_DATA_PATH.parent}")
