"""Unit tests for EPred data loader and preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    BASE_TARGET_COL,
    FEATURE_ORDER,
    REQUIRED_RAW_COLUMNS,
)
from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline


def test_data_loader_validates_raw_columns():
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in REQUIRED_RAW_COLUMNS:
        assert col in df.columns, f"Required column {col} missing in raw data."


def test_data_loader_chronological_ordering():
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    # Ensure timestamps are strictly non-decreasing
    assert df["date"].is_monotonic_increasing, "Dataset must be sorted chronologically."


def test_data_loader_inspection_diagnostics():
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    diagnostics = loader.inspect_data(df)
    assert diagnostics["total_rows"] == len(df)
    assert diagnostics["duplicate_rows"] == 0
    assert diagnostics["duplicate_timestamps"] == 0
    assert diagnostics["total_missing_values"] == 0
    assert diagnostics["regular_intervals_count"] > 0


def test_target_generation_no_future_leakage():
    pipeline = PreprocessingPipeline(RAW_DATA_PATH)
    df, meta = pipeline.prepare_dataset()

    assert TARGET_COL in df.columns
    # Ensure no NaN remains in target
    assert not df[TARGET_COL].isnull().any()
    # Confirm target is exactly 10 min ahead
    # For any row i, target must equal BASE_TARGET_COL of the raw data's next step
    raw_df = pipeline.data_loader.load_data()
    # Due to dropping first 12 warmup rows:
    # row 0 of prepared df corresponds to row 12 of raw_df
    # and its target must match row 13's Appliances
    assert df[TARGET_COL].iloc[0] == raw_df[BASE_TARGET_COL].iloc[13]


def test_chronological_splitting_bounds():
    pipeline = PreprocessingPipeline(RAW_DATA_PATH)
    df, _ = pipeline.prepare_dataset()
    train_df, val_df, test_df = pipeline.chronological_split(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Strictly verify chronological separation: train < val < test
    assert train_df["date"].max() < val_df["date"].min(), "Train timestamps must precede Validation."
    assert val_df["date"].max() < test_df["date"].min(), "Validation timestamps must precede Test."


def test_scaler_fitted_on_training_data_only():
    pipeline = PreprocessingPipeline(RAW_DATA_PATH)
    df, _ = pipeline.prepare_dataset()
    train_df, val_df, _ = pipeline.chronological_split(df)

    scaler = pipeline.fit_scaler(train_df[FEATURE_ORDER])
    assert scaler.mean_ is not None
    assert len(scaler.mean_) == len(FEATURE_ORDER)

    # Transform training and val
    X_train_scaled = pipeline.transform_features(train_df)
    X_val_scaled = pipeline.transform_features(val_df)

    # Scaled training data should have zero mean and unit variance approximately
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-3)
    assert np.allclose(np.var(X_train_scaled, axis=0), 1, atol=1e-3)

    # Validation data will NOT have exact 0 mean because it was not used to fit the scaler!
    assert not np.allclose(np.mean(X_val_scaled, axis=0), 0, atol=1e-3)
