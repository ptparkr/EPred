"""Data loader module for EPred.

Responsible for loading the raw UCI Appliances Energy dataset, validating schema,
parsing timestamps, ensuring chronological sorting, checking duplicates, missing values,
invalid numeric values, and checking timestamp intervals.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH, REQUIRED_RAW_COLUMNS

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles raw data loading and schema integrity checks."""

    def __init__(self, file_path: Path = RAW_DATA_PATH):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """Loads raw dataset from disk and performs initial integrity checks.

        Returns:
            pd.DataFrame: Cleaned, sorted DataFrame with parsed datetime.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If required columns are missing or data is malformed.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.file_path}. Please place "
                "energydata_complete.csv into data/raw/ directory."
            )

        logger.info(f"Loading raw dataset from {self.file_path}...")
        df = pd.read_csv(self.file_path)

        # 1. Validate required columns
        missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # 2. Parse date into datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isnull().any():
            invalid_dates = df[df["date"].isnull()]
            raise ValueError(f"Found {len(invalid_dates)} unparseable date values.")

        # 3. Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def inspect_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspects data quality, duplicates, missing values, and timestamp intervals.

        Args:
            df (pd.DataFrame): Input DataFrame with 'date' as datetime.

        Returns:
            Dict[str, Any]: Diagnostic report dictionary.
        """
        num_rows, num_cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        duplicate_timestamps = int(df["date"].duplicated().sum())

        missing_counts = df.isnull().sum().to_dict()
        total_missing = int(sum(missing_counts.values()))

        # Check for inf / -inf in numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        for col in numeric_cols:
            inf_sum = int(np.isinf(df[col]).sum())
            if inf_sum > 0:
                infinite_counts[col] = inf_sum

        # Check timestamp intervals
        time_diffs = df["date"].diff().dropna()
        interval_counts = time_diffs.value_counts().to_dict()
        # Convert Timedelta keys to string for readability
        interval_distribution = {str(k): int(v) for k, v in interval_counts.items()}
        expected_10min = pd.Timedelta(minutes=10)
        regular_interval_count = int((time_diffs == expected_10min).sum())
        irregular_intervals = int(len(time_diffs) - regular_interval_count)

        report = {
            "total_rows": num_rows,
            "total_columns": num_cols,
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max()),
            "duplicate_rows": duplicate_rows,
            "duplicate_timestamps": duplicate_timestamps,
            "total_missing_values": total_missing,
            "infinite_values": infinite_counts,
            "expected_interval": "10 minutes",
            "regular_intervals_count": regular_interval_count,
            "irregular_intervals_count": irregular_intervals,
            "interval_distribution": interval_distribution,
        }

        logger.info(f"Dataset inspection complete: {report}")
        return report


def load_raw_dataset(file_path: Path = RAW_DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to load and inspect raw data.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Clean loaded data and inspection report.
    """
    loader = DataLoader(file_path=file_path)
    df = loader.load_data()
    diagnostics = loader.inspect_data(df)
    return df, diagnostics
