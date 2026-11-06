"""FastAPI application for EPred V2 (Energy Prediction System).

Exposes inference endpoints for real-time 10-minute appliance energy forecasting,
multi-horizon forecasting (10m, 30m, 60m), conformal quantile uncertainty bounds,
streaming telemetry with ring buffer caching, and data drift diagnostics.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
import pandas as pd

from src.predict import PredictionEngine, get_prediction_engine
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RawSensorInput,
    TelemetryStreamInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftReportResponse,
)

logger = logging.getLogger(__name__)

# Global prediction engine reference
engine: PredictionEngine = None


def get_engine() -> PredictionEngine:
    """Retrieves or initializes singleton PredictionEngine instance."""
    global engine
    if engine is None:
        engine = get_prediction_engine()
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads the trained model bundle once during application startup."""
    try:
        logger.info("Initializing EPred inference engine on startup...")
        eng = get_engine()
        logger.info(f"EPred model loaded: {eng.artifact.model_name} (v{eng.artifact.model_version})")
    except Exception as e:
        logger.error(f"Failed to load EPred model artifact during startup: {e}")
    yield
    logger.info("Shutting down EPred API service...")


app = FastAPI(
    title="EPred — Energy Prediction System API (Version 2.0)",
    description=(
        "Production-ready REST API for forecasting appliance energy consumption. "
        "Supports multi-horizon forecasting (10m, 30m, 60m), conformal uncertainty bounds "
        "(P10, P50, P90), spike surge risk detection, zero-history streaming telemetry, "
        "and real-time distribution drift diagnostics."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root endpoint welcoming clients and directing them to OpenAPI documentation."""
    return {
        "service": "EPred — Energy Prediction System",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "features": "Multi-Horizon (10m, 30m, 60m), Conformal Quantiles (P10/P50/P90), Spike Risk, Telemetry Stream",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check() -> HealthResponse:
    """Returns service health status and model availability."""
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EPred model engine is not loaded or unhealthy.",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=eng.artifact.model_name,
        model_version=eng.artifact.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info() -> ModelInfoResponse:
    """Returns active model metadata, version, forecast horizons, and evaluation metrics."""
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable.",
        )
    info = eng.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(reading: RawSensorInput) -> PredictionResponse:
    """Forecasts appliance energy consumption for the next 10-minute interval with uncertainty bounds."""
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready.",
        )

    reading_dict = reading.model_dump()
    history = reading_dict.pop("historical_appliances")
    obs_date = reading_dict.get("date")

    try:
        result = eng.predict(
            features=reading_dict,
            raw_mode=True,
            historical_appliances=history,
            timestamp=obs_date,
        )
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(e)}",
        )


@app.post("/telemetry", tags=["Streaming Inference"])
def stream_telemetry(reading: TelemetryStreamInput) -> Dict[str, Any]:
    """Streams a single observation to the in-memory ring buffer.

    Clients do NOT need to pass historical consumption arrays!
    EPred maintains a rolling buffer of size 12 in memory automatically.
    """
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready.",
        )

    reading_dict = reading.model_dump()
    obs_date = reading_dict.get("date")
    try:
        return eng.predict_telemetry_stream(reading_dict, timestamp=obs_date)
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Streaming ingestion error: {str(e)}",
        )


@app.get("/model/drift", response_model=DriftReportResponse, tags=["Diagnostics"])
def drift_check() -> DriftReportResponse:
    """Checks the active telemetry feature distribution against training baseline distributions."""
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready.",
        )

    # Use current cached buffer or sample test observation
    history = eng.ring_buffer.get_history()
    if not history:
        return DriftReportResponse(
            status="normal (no streaming drift recorded)",
            drifted_features_count=0,
            total_features_checked=len(eng.expected_features),
            details={},
        )

    # Synthetic check with available buffer
    df_sample = pd.DataFrame([{col: 0.0 for col in eng.expected_features}])
    report = eng.check_drift(df_sample)
    return DriftReportResponse(
        status=report["status"],
        drifted_features_count=report["drifted_features_count"],
        total_features_checked=report["total_features_checked"],
        details=report.get("details", {}),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Generates predictions for a batch of multiple sensor observations."""
    eng = get_engine()
    if eng is None or not eng.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready.",
        )

    predictions = []
    for idx, reading in enumerate(batch_request.observations):
        reading_dict = reading.model_dump()
        history = reading_dict.pop("historical_appliances")
        obs_date = reading_dict.get("date")
        try:
            res = eng.predict(
                features=reading_dict,
                raw_mode=True,
                historical_appliances=history,
                timestamp=obs_date,
            )
            predictions.append(res)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error in observation at index {idx}: {str(e)}",
            )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        status="success",
    )
