"""Integration and validation tests for EPred V2 FastAPI endpoints and Pydantic schemas.

Tests endpoint logic, schema validations, error responses, streaming telemetry,
and health/drift diagnostics.
"""

import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from api.main import (
    root,
    health_check,
    model_info,
    predict_single,
    predict_batch,
    stream_telemetry,
    drift_check,
)
from api.schemas import (
    RawSensorInput,
    TelemetryStreamInput,
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    BatchPredictionResponse,
    DriftReportResponse,
)
from src.predict import get_prediction_engine
import api.main as api_main

# Ensure global engine is initialized for tests
api_main.engine = get_prediction_engine()


@pytest.fixture
def valid_sensor_dict():
    """Provides valid payload dictionary including room sensors and 12 historical readings."""
    return {
        "date": "2016-01-11 17:00:00",
        "Appliances": 60.0,
        "lights": 20.0,
        "T1": 19.89, "RH_1": 47.59,
        "T2": 19.2, "RH_2": 44.79,
        "T3": 19.79, "RH_3": 44.73,
        "T4": 19.0, "RH_4": 45.56,
        "T5": 17.16, "RH_5": 55.2,
        "T6": 7.02, "RH_6": 84.26,
        "T7": 17.2, "RH_7": 41.62,
        "T8": 18.2, "RH_8": 48.9,
        "T9": 17.03, "RH_9": 45.53,
        "T_out": 6.6, "Press_mm_hg": 733.5, "RH_out": 92.0,
        "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3,
        "historical_appliances": [60.0, 60.0, 50.0, 50.0, 60.0, 60.0, 50.0, 60.0, 50.0, 40.0, 50.0, 60.0]
    }


def test_root_endpoint():
    res = root()
    assert res["service"] == "EPred — Energy Prediction System"
    assert "/docs" in res["docs_url"]
    assert res["version"] == "2.0.0"


def test_health_endpoint():
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.model_version == "2.0.0"


def test_model_info_endpoint():
    res = model_info()
    assert isinstance(res, ModelInfoResponse)
    assert res.forecast_horizon_minutes == 10
    assert res.supported_horizons_minutes == [10, 30, 60]
    assert res.num_features == 48
    assert "RMSE" in res.test_metrics
    assert "p10_offset" in res.conformal_offsets


def test_predict_endpoint_valid_payload(valid_sensor_dict):
    input_obj = RawSensorInput(**valid_sensor_dict)
    res = predict_single(input_obj)

    assert isinstance(res, PredictionResponse)
    assert res.status == "success"
    assert res.unit == "Wh"
    assert res.forecast_horizon_minutes == 10
    assert res.predicted_appliances_wh >= 0.0

    # V2 Enhancements verification
    assert res.confidence_intervals is not None
    ci = res.confidence_intervals
    assert ci.p10_wh <= ci.p50_median_wh
    assert ci.p50_median_wh <= ci.p90_wh

    assert res.multi_horizon_forecast is not None
    assert "10min" in res.multi_horizon_forecast
    assert "30min" in res.multi_horizon_forecast
    assert "60min" in res.multi_horizon_forecast

    assert res.spike_risk is not None
    assert 0.0 <= res.spike_risk.probability <= 1.0


def test_streaming_telemetry_endpoint(valid_sensor_dict):
    telemetry_dict = valid_sensor_dict.copy()
    telemetry_dict.pop("historical_appliances")

    input_obj = TelemetryStreamInput(**telemetry_dict)
    eng = api_main.get_engine()
    eng.ring_buffer.clear()

    # Streaming 12 warmup readings
    for _ in range(12):
        stream_telemetry(input_obj)

    res13 = stream_telemetry(input_obj)
    assert res13["status"] == "success"
    assert res13["predicted_appliances_wh"] >= 0.0


def test_drift_endpoint():
    res = drift_check()
    assert isinstance(res, DriftReportResponse)
    assert "total_features_checked" in res.model_dump()


def test_predict_endpoint_insufficient_history(valid_sensor_dict):
    bad_dict = valid_sensor_dict.copy()
    bad_dict["historical_appliances"] = [60.0] * 8  # Only 8 readings

    with pytest.raises(ValidationError) as exc_info:
        RawSensorInput(**bad_dict)
    assert "at least 12 values" in str(exc_info.value)


def test_predict_endpoint_negative_consumption(valid_sensor_dict):
    bad_dict = valid_sensor_dict.copy()
    bad_dict["Appliances"] = -25.0

    with pytest.raises(ValidationError) as exc_info:
        RawSensorInput(**bad_dict)
    assert "greater than or equal to 0" in str(exc_info.value)


def test_predict_batch_endpoint(valid_sensor_dict):
    input_obj = RawSensorInput(**valid_sensor_dict)
    batch_req = BatchPredictionRequest(observations=[input_obj, input_obj])
    res = predict_batch(batch_req)

    assert isinstance(res, BatchPredictionResponse)
    assert res.status == "success"
    assert res.count == 2
    assert len(res.predictions) == 2
    for pred in res.predictions:
        assert pred["predicted_appliances_wh"] >= 0.0
