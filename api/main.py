"""FastAPI application for EPred (Energy Prediction System).

Exposes inference endpoints for real-time 10-minute appliance energy forecasting.
Ensures zero future-data leakage, model caching on startup, and rigorous Pydantic validation.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.predict import PredictionEngine, get_prediction_engine
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RawSensorInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)

# Global prediction engine reference
engine: PredictionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads the trained model bundle once during application startup."""
    global engine
    try:
        logger.info("Initializing EPred inference engine on startup...")
        engine = get_prediction_engine()
        logger.info(f"EPred model loaded: {engine.artifact.model_name} (v{engine.artifact.model_version})")
    except Exception as e:
        logger.error(f"Failed to load EPred model artifact during startup: {e}")
        engine = None
    yield
    logger.info("Shutting down EPred API service...")


app = FastAPI(
    title="EPred — Energy Prediction System API",
    description=(
        "Production-ready REST API for forecasting appliance energy consumption "
        "for the next 10-minute interval using room-specific environmental sensors, "
        "outdoor weather measurements, and historical energy consumption."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root endpoint welcoming clients and directing them to OpenAPI documentation."""
    return {
        "service": "EPred — Energy Prediction System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "description": "Forecasts appliance energy consumption (Wh) for the next 10-minute interval.",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check() -> HealthResponse:
    """Returns service health status and model availability."""
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EPred model engine is not loaded or unhealthy.",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=engine.artifact.model_name,
        model_version=engine.artifact.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info() -> ModelInfoResponse:
    """Returns active model metadata, version, forecast horizon, and evaluation metrics."""
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable.",
        )
    info = engine.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(reading: RawSensorInput) -> PredictionResponse:
    """Forecasts appliance energy consumption for the next 10-minute interval.

    Requires:
    - Room-specific temperature and humidity sensors (T1-T9, RH_1-RH_9)
    - Outside weather station measurements (T_out, Press_mm_hg, etc.)
    - Current appliance reading (Appliances) and lighting (lights)
    - Chronological historical appliance consumption (at least 12 preceding 10-min readings)
    """
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready.",
        )

    reading_dict = reading.model_dump()
    history = reading_dict.pop("historical_appliances")
    obs_date = reading_dict.get("date")

    try:
        result = engine.predict(
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


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Generates predictions for a batch of multiple sensor observations."""
    if engine is None or not engine.is_healthy():
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
            res = engine.predict(
                features=reading_dict,
                raw_mode=True,
                historical_appliances=history,
                timestamp=obs_date,
            )
            predictions.append(PredictionResponse(**res))
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
