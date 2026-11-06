"""Unit tests for EPred V2 model training, evaluation, persistence, and prediction engine."""

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
    # V2 Enhancements
    assert "30min" in artifact.multi_horizon_estimators
    assert "60min" in artifact.multi_horizon_estimators
    assert "p10_offset" in artifact.quantile_offsets
    assert "p90_offset" in artifact.quantile_offsets
    assert artifact.spike_classifier is not None
    assert len(artifact.feature_baselines) == len(FEATURE_ORDER)


def test_metrics_calculation_known_values():
    """Validates regression metric formula outputs with known analytical values."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])  # Errors: +10, -10, 0

    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 6.6667, atol=1e-3)
    assert np.isclose(metrics["MSE"], 66.6667, atol=1e-3)
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


def test_conformal_quantiles_and_multi_horizon():
    """Verifies V2 conformal uncertainty intervals and multi-horizon trajectory."""
    engine = get_prediction_engine()
    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_features = test_df.iloc[2:3]

    result = engine.predict(sample_features)
    ci = result["confidence_intervals"]
    assert ci["p10_wh"] <= ci["p50_median_wh"]
    assert ci["p50_median_wh"] <= ci["p90_wh"]

    # Multi-horizon trajectory
    traj = result["multi_horizon_forecast"]
    assert "10min" in traj
    assert "30min" in traj
    assert "60min" in traj
    assert traj["10min"] >= 0.0
    assert traj["30min"] >= 0.0
    assert traj["60min"] >= 0.0

    # Spike risk assessment
    spike = result["spike_risk"]
    assert 0.0 <= spike["probability"] <= 1.0
    assert spike["risk_level"] in ["low", "medium", "high"]


def test_telemetry_ring_buffer_streaming():
    """Verifies in-memory ring buffer enables streaming without client historical array."""
    engine = PredictionEngine()
    engine.ring_buffer.clear()

    raw_reading = {
        "Appliances": 60.0, "lights": 10.0,
        "T1": 20.0, "RH_1": 45.0, "T2": 19.5, "RH_2": 44.0, "T3": 20.2, "RH_3": 43.5,
        "T4": 19.8, "RH_4": 42.0, "T5": 18.5, "RH_5": 50.0, "T6": 8.5, "RH_6": 80.0,
        "T7": 18.0, "RH_7": 40.0, "T8": 19.0, "RH_8": 45.0, "T9": 18.2, "RH_9": 43.0,
        "T_out": 7.0, "Press_mm_hg": 740.0, "RH_out": 85.0, "Windspeed": 5.0,
        "Visibility": 50.0, "Tdewpoint": 4.5
    }

    # First 12 calls warm up the buffer with 12 historical readings
    for i in range(12):
        res = engine.predict_telemetry_stream(raw_reading)
        assert res["status"] == "warming_up"
        assert res["buffer_size"] == i + 1

    # 13th call has 12 historical readings ready and successfully produces forecast!
    res13 = engine.predict_telemetry_stream(raw_reading)
    assert res13["status"] == "success"
    assert "predicted_appliances_wh" in res13
    assert res13["predicted_appliances_wh"] >= 0.0


def test_drift_detection():
    """Verifies distribution drift checking against training baselines."""
    engine = get_prediction_engine()
    test_df = pd.read_csv(TEST_DATA_PATH)

    normal_sample = test_df.iloc[0:5][FEATURE_ORDER]
    report_normal = engine.check_drift(normal_sample)
    assert report_normal["total_features_checked"] == len(FEATURE_ORDER)
    assert report_normal["status"] in ["normal", "minor_drift"]

    # Injected extreme anomaly sample
    drifted_sample = normal_sample.copy()
    drifted_sample["T1"] = 999.0  # 999 Celsius in kitchen
    drifted_sample["Appliances"] = 9999.0

    report_drift = engine.check_drift(drifted_sample)
    assert report_drift["drifted_features_count"] >= 2
    assert report_drift["details"]["T1"]["drift_flag"] is True
