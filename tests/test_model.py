"""Unit tests for EPred model training, evaluation, persistence, and prediction engine."""

import pytest
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, TEST_DATA_PATH, FEATURE_ORDER
from src.model_registry import load_model_artifact, ModelArtifact
from src.predict import PredictionEngine, get_prediction_engine
from src.evaluate import calculate_metrics


def test_model_artifact_integrity():
    """Ensures the persisted model artifact has all mandatory components."""
    artifact = load_model_artifact(BEST_MODEL_PATH)

    assert isinstance(artifact, ModelArtifact)
    assert artifact.estimator is not None
    assert artifact.scaler is not None
    assert len(artifact.feature_order) == len(FEATURE_ORDER)
    assert artifact.model_name in ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    assert artifact.forecast_horizon_minutes == 10
    assert "RMSE" in artifact.val_metrics
    assert "RMSE" in artifact.test_metrics
    assert artifact.val_metrics["RMSE"] > 0
    assert artifact.test_metrics["RMSE"] > 0


def test_metrics_calculation_known_values():
    """Validates regression metric formula outputs with known analytical values."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])  # Errors: +10, -10, 0

    metrics = calculate_metrics(y_true, y_pred)
    # MAE = (10 + 10 + 0) / 3 = 6.6667
    assert np.isclose(metrics["MAE"], 6.6667, atol=1e-3)
    # MSE = (100 + 100 + 0) / 3 = 66.6667
    assert np.isclose(metrics["MSE"], 66.6667, atol=1e-3)
    # RMSE = sqrt(66.6667) = 8.165
    assert np.isclose(metrics["RMSE"], 8.165, atol=1e-3)


def test_prediction_engine_single_and_finite():
    """Verifies that PredictionEngine outputs finite, non-negative predictions matching schema."""
    engine = get_prediction_engine()
    assert engine.is_healthy()

    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_features = test_df.iloc[0:1]

    result = engine.predict(sample_features)
    assert result["status"] == "success"
    assert result["unit"] == "Wh"
    assert result["forecast_horizon_minutes"] == 10
    assert np.isfinite(result["predicted_appliances_wh"])
    assert result["predicted_appliances_wh"] >= 0.0


def test_prediction_consistency():
    """Verifies deterministic inference: duplicate requests produce identical predictions."""
    engine = get_prediction_engine()
    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_features = test_df.iloc[5:6]

    res1 = engine.predict(sample_features)
    res2 = engine.predict(sample_features)
    assert res1["predicted_appliances_wh"] == res2["predicted_appliances_wh"]


def test_prediction_batch_accuracy():
    """Verifies batch inference matches sequential single predictions."""
    engine = get_prediction_engine()
    test_df = pd.read_csv(TEST_DATA_PATH)
    batch_features = test_df.iloc[10:15]

    batch_res = engine.predict_batch(batch_features)
    assert len(batch_res) == 5

    for i in range(5):
        single_res = engine.predict(batch_features.iloc[i:i+1])
        assert batch_res[i]["predicted_appliances_wh"] == single_res["predicted_appliances_wh"]


def test_prediction_engine_raw_mode_validation():
    """Verifies raw observation calculation and lag validation."""
    engine = get_prediction_engine()
    test_df = pd.read_csv(TEST_DATA_PATH)
    row = test_df.iloc[20].to_dict()

    valid_history = [50.0] * 12
    res = engine.predict(row, raw_mode=True, historical_appliances=valid_history)
    assert res["status"] == "success"
    assert np.isfinite(res["predicted_appliances_wh"])

    # Test error on insufficient history (< 12)
    short_history = [50.0] * 10
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.predict(row, raw_mode=True, historical_appliances=short_history)
