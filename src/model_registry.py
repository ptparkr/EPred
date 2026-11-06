"""Model registry and persistence module for EPred.

Responsible for bundling trained estimators, multi-horizon heads, conformal quantile calibration,
feature baselines for drift detection, serializing via Joblib, and providing deserialization.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_VERSION,
    FORECAST_HORIZON_MINUTES,
    SUPPORTED_HORIZONS_MINUTES,
    FEATURE_ORDER,
)

logger = logging.getLogger(__name__)


class ModelArtifact:
    """Encapsulates the complete trained model bundle required for inference and diagnostics."""

    def __init__(
        self,
        estimator: BaseEstimator,
        scaler: StandardScaler,
        feature_order: List[str],
        model_name: str,
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
        model_version: str = MODEL_VERSION,
        forecast_horizon_minutes: int = FORECAST_HORIZON_MINUTES,
        multi_horizon_estimators: Optional[Dict[str, BaseEstimator]] = None,
        quantile_offsets: Optional[Dict[str, float]] = None,
        spike_classifier: Optional[BaseEstimator] = None,
        feature_baselines: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        training_timestamp: Optional[str] = None,
    ):
        self.estimator = estimator
        self.scaler = scaler
        self.feature_order = feature_order
        self.model_name = model_name
        self.model_version = model_version
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics or {}
        self.forecast_horizon_minutes = forecast_horizon_minutes
        self.multi_horizon_estimators = multi_horizon_estimators or {}
        self.quantile_offsets = quantile_offsets or {"p10_offset": -20.0, "p90_offset": 45.0}
        self.spike_classifier = spike_classifier
        self.feature_baselines = feature_baselines or {}
        self.dataset_info = dataset_info or {}
        self.training_timestamp = training_timestamp or datetime.now(timezone.utc).isoformat()

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Exports human-readable JSON-serializable metadata."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "forecast_horizon_minutes": self.forecast_horizon_minutes,
            "supported_horizons_minutes": SUPPORTED_HORIZONS_MINUTES,
            "training_timestamp": self.training_timestamp,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "conformal_quantile_offsets": self.quantile_offsets,
            "has_spike_classifier": self.spike_classifier is not None,
            "multi_horizon_heads": list(self.multi_horizon_estimators.keys()),
            "dataset_info": self.dataset_info,
            "num_features": len(self.feature_order),
            "feature_order": self.feature_order,
        }


def save_model_artifact(
    artifact: ModelArtifact,
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> None:
    """Serializes the ModelArtifact to disk via Joblib and exports metadata JSON."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_path)
    logger.info(f"Model artifact saved to {model_path}")

    metadata = artifact.to_metadata_dict()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata exported to {metadata_path}")


def load_model_artifact(model_path: Path = BEST_MODEL_PATH) -> ModelArtifact:
    """Deserializes and validates a saved ModelArtifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at {model_path}. Train the model first."
        )

    artifact = joblib.load(model_path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(
            f"Loaded object is of type {type(artifact)}, expected ModelArtifact."
        )

    logger.info(
        f"Loaded model '{artifact.model_name}' (version: {artifact.model_version}) "
        f"trained at {artifact.training_timestamp}"
    )
    return artifact
