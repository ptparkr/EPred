"""Integration and validation tests for EPred FastAPI endpoints and Pydantic schemas.

Tests endpoint logic, schema validations, error responses, and health diagnostics.
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
)
from api.schemas import (
    RawSensorInput,
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    BatchPredictionResponse,
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


def test_health_endpoint():
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.model_version == "1.0.0"


def test_model_info_endpoint():
    res = model_info()
    assert isinstance(res, ModelInfoResponse)
    assert res.forecast_horizon_minutes == 10
    assert res.num_features == 48
    assert "RMSE" in res.test_metrics
    assert res.test_metrics["RMSE"] > 0


def test_predict_endpoint_valid_payload(valid_sensor_dict):
    input_obj = RawSensorInput(**valid_sensor_dict)
    res = predict_single(input_obj)

    assert isinstance(res, PredictionResponse)
    assert res.status == "success"
    assert res.unit == "Wh"
    assert res.forecast_horizon_minutes == 10
    assert res.predicted_appliances_wh >= 0.0


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
        assert pred.predicted_appliances_wh >= 0.0
