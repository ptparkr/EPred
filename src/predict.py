"""Prediction engine module for EPred.

Provides a unified, validated interface for single and batch predictions.
Supports both Python function calls and REST API endpoints.
Enforces feature ordering, validation, and zero future-data leakage.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_ORDER,
    FORECAST_HORIZON_MINUTES,
    MODEL_VERSION,
    INDOOR_TEMP_COLS,
    INDOOR_HUMIDITY_COLS,
    OUTDOOR_BUILDING_TEMP_COL,
    OUTDOOR_BUILDING_HUMIDITY_COL,
)
from src.model_registry import ModelArtifact, load_model_artifact

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Production inference engine for EPred."""

    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        self.artifact: Optional[ModelArtifact] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads and verifies model artifact from storage."""
        self.artifact = load_model_artifact(self.model_path)
        self.expected_features = self.artifact.feature_order

    def is_healthy(self) -> bool:
        """Returns True if the engine is initialized and ready for inference."""
        return self.artifact is not None and self.artifact.estimator is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active inference model."""
        if not self.is_healthy():
            raise RuntimeError("Prediction engine model is not loaded.")
        return {
            "model_name": self.artifact.model_name,
            "model_version": self.artifact.model_version,
            "forecast_horizon_minutes": self.artifact.forecast_horizon_minutes,
            "training_timestamp": self.artifact.training_timestamp,
            "num_features": len(self.expected_features),
            "validation_metrics": self.artifact.val_metrics,
            "test_metrics": self.artifact.test_metrics,
        }

    def compute_features_from_raw(
        self,
        raw_reading: Dict[str, Any],
        historical_appliances: List[float],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, float]:
        """Derives all required features from raw sensor readings and historical consumption.

        Args:
            raw_reading: Dictionary containing current interval sensor measurements.
                         Must include 'Appliances', 'lights', T1-T9, RH_1-RH_9,
                         T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint.
            historical_appliances: List of preceding completed appliance consumption readings
                                   (at least 12 readings in chronological order [t-12, ..., t-1]).
            timestamp: Datetime or ISO timestamp of the current observation.
                       Defaults to raw_reading.get('date') or current UTC time.

        Returns:
            Dict[str, float]: Fully engineered feature map.
        """
        # 1. Validate historical consumption sequence
        if not historical_appliances or len(historical_appliances) < 12:
            raise ValueError(
                f"Insufficient historical data: received {len(historical_appliances) if historical_appliances else 0} "
                "historical readings. Exactly or at least 12 past 10-minute readings are required to compute lag and rolling features."
            )

        # Ensure finite numeric values in history
        history_arr = np.asarray(historical_appliances, dtype=float)
        if not np.all(np.isfinite(history_arr)):
            raise ValueError("Historical appliances series contains NaN or infinite values.")

        # 2. Validate current reading
        curr_appliances = float(raw_reading.get("Appliances", 0))
        if not np.isfinite(curr_appliances) or curr_appliances < 0:
            raise ValueError(f"Invalid current Appliances value: {curr_appliances}. Must be non-negative finite number.")

        # 3. Lags: historical readings ordered from oldest to most recent [t-12, ..., t-1]
        lag_1 = float(history_arr[-1])
        lag_2 = float(history_arr[-2])
        lag_3 = float(history_arr[-3])
        lag_6 = float(history_arr[-6])
        lag_12 = float(history_arr[-12])

        # Rolling statistics including current observation (available at or before prediction time t)
        window_3 = np.append(history_arr[-2:], curr_appliances)
        window_6 = np.append(history_arr[-5:], curr_appliances)
        window_12 = np.append(history_arr[-11:], curr_appliances)

        roll_mean_3 = float(np.mean(window_3))
        roll_mean_6 = float(np.mean(window_6))
        roll_mean_12 = float(np.mean(window_12))
        roll_std_6 = float(np.std(window_6, ddof=1)) if len(window_6) > 1 else 0.0

        # 4. Temporal features
        ts_val = timestamp or raw_reading.get("date", datetime.now(timezone.utc))
        dt = pd.to_datetime(ts_val)
        hour = int(dt.hour)
        minute = int(dt.minute)
        day_of_week = int(dt.day_of_week)
        is_weekend = int(day_of_week >= 5)
        month = int(dt.month)

        continuous_hour = hour + minute / 60.0
        hour_sin = float(np.sin(2.0 * np.pi * continuous_hour / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * continuous_hour / 24.0))
        weekday_sin = float(np.sin(2.0 * np.pi * day_of_week / 7.0))
        weekday_cos = float(np.cos(2.0 * np.pi * day_of_week / 7.0))

        # 5. Environmental indoor averages and gradients
        try:
            in_temps = [float(raw_reading[k]) for k in INDOOR_TEMP_COLS]
            in_hums = [float(raw_reading[k]) for k in INDOOR_HUMIDITY_COLS]
            out_temp = float(raw_reading[OUTDOOR_BUILDING_TEMP_COL])
            out_hum = float(raw_reading[OUTDOOR_BUILDING_HUMIDITY_COL])
        except KeyError as e:
            raise ValueError(f"Missing required sensor measurement in raw reading: {e}")

        avg_in_temp = float(np.mean(in_temps))
        avg_in_hum = float(np.mean(in_hums))
        temp_diff = float(avg_in_temp - out_temp)
        hum_diff = float(avg_in_hum - out_hum)

        # Assemble full feature map
        feature_dict = {
            "Appliances": curr_appliances,
            "lights": float(raw_reading.get("lights", 0.0)),
            "T1": float(raw_reading["T1"]),
            "RH_1": float(raw_reading["RH_1"]),
            "T2": float(raw_reading["T2"]),
            "RH_2": float(raw_reading["RH_2"]),
            "T3": float(raw_reading["T3"]),
            "RH_3": float(raw_reading["RH_3"]),
            "T4": float(raw_reading["T4"]),
            "RH_4": float(raw_reading["RH_4"]),
            "T5": float(raw_reading["T5"]),
            "RH_5": float(raw_reading["RH_5"]),
            "T6": float(raw_reading["T6"]),
            "RH_6": float(raw_reading["RH_6"]),
            "T7": float(raw_reading["T7"]),
            "RH_7": float(raw_reading["RH_7"]),
            "T8": float(raw_reading["T8"]),
            "RH_8": float(raw_reading["RH_8"]),
            "T9": float(raw_reading["T9"]),
            "RH_9": float(raw_reading["RH_9"]),
            "T_out": float(raw_reading["T_out"]),
            "Press_mm_hg": float(raw_reading["Press_mm_hg"]),
            "RH_out": float(raw_reading["RH_out"]),
            "Windspeed": float(raw_reading["Windspeed"]),
            "Visibility": float(raw_reading["Visibility"]),
            "Tdewpoint": float(raw_reading["Tdewpoint"]),
            "Hour": hour,
            "Minute": minute,
            "Day_of_Week": day_of_week,
            "Is_Weekend": is_weekend,
            "Month": month,
            "Hour_Sin": hour_sin,
            "Hour_Cos": hour_cos,
            "Weekday_Sin": weekday_sin,
            "Weekday_Cos": weekday_cos,
            "Appliances_Lag_1": lag_1,
            "Appliances_Lag_2": lag_2,
            "Appliances_Lag_3": lag_3,
            "Appliances_Lag_6": lag_6,
            "Appliances_Lag_12": lag_12,
            "Appliances_Rolling_Mean_3": roll_mean_3,
            "Appliances_Rolling_Mean_6": roll_mean_6,
            "Appliances_Rolling_Mean_12": roll_mean_12,
            "Appliances_Rolling_Std_6": roll_std_6,
            "Avg_Indoor_Temp": avg_in_temp,
            "Avg_Indoor_Humidity": avg_in_hum,
            "Temp_Difference": temp_diff,
            "Humidity_Difference": hum_diff,
        }

        return feature_dict

    def _validate_and_order_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Validates feature names, checks for missing/infinite values, and guarantees strict column order."""
        missing = [f for f in self.expected_features if f not in df_features.columns]
        if missing:
            raise ValueError(f"Input features missing required columns: {missing}")

        ordered_df = df_features[self.expected_features].copy()

        # Check for NaN / Inf
        if ordered_df.isnull().any().any():
            nan_cols = ordered_df.columns[ordered_df.isnull().any()].tolist()
            raise ValueError(f"Input features contain NaN values in columns: {nan_cols}")

        for col in ordered_df.columns:
            if not np.all(np.isfinite(ordered_df[col])):
                raise ValueError(f"Input column '{col}' contains infinite values.")

        return ordered_df

    def predict(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
        raw_mode: bool = False,
        historical_appliances: Optional[List[float]] = None,
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Generates single next-10-minute prediction.

        Args:
            features: Dictionary of engineered features or raw measurements.
            raw_mode: If True, computes engineered features using raw_reading and historical_appliances.
            historical_appliances: Required if raw_mode is True.
            timestamp: Datetime of the observation.

        Returns:
            Dict[str, Any]: Prediction response with predicted consumption, units, metadata.
        """
        if not self.is_healthy():
            raise RuntimeError("Model is not initialized.")

        if raw_mode:
            feature_dict = self.compute_features_from_raw(
                features, historical_appliances, timestamp
            )
            df_in = pd.DataFrame([feature_dict])
        elif isinstance(features, dict):
            df_in = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            df_in = features.copy()
        else:
            raise TypeError("Features must be a dictionary or pandas DataFrame.")

        ordered_df = self._validate_and_order_features(df_in)

        # Scale features if scaler exists
        if self.artifact.scaler is not None:
            X = self.artifact.scaler.transform(ordered_df)
        else:
            X = ordered_df.values

        raw_pred = float(self.artifact.estimator.predict(X)[0])
        # Energy consumption in Wh is non-negative
        predicted_wh = max(0.0, round(raw_pred, 2))

        pred_time = timestamp or datetime.now(timezone.utc).isoformat()
        if isinstance(pred_time, datetime):
            pred_time = pred_time.isoformat()

        return {
            "predicted_appliances_wh": predicted_wh,
            "unit": "Wh",
            "forecast_horizon_minutes": self.artifact.forecast_horizon_minutes,
            "prediction_timestamp": str(pred_time),
            "model_version": self.artifact.model_version,
            "model_name": self.artifact.model_name,
            "status": "success",
        }

    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generates predictions for a batch of pre-engineered feature rows."""
        if not self.is_healthy():
            raise RuntimeError("Model is not initialized.")

        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame.")

        ordered_df = self._validate_and_order_features(features_df)

        if self.artifact.scaler is not None:
            X = self.artifact.scaler.transform(ordered_df)
        else:
            X = ordered_df.values

        raw_preds = self.artifact.estimator.predict(X)

        results = []
        for i, val in enumerate(raw_preds):
            predicted_wh = max(0.0, round(float(val), 2))
            results.append({
                "index": i,
                "predicted_appliances_wh": predicted_wh,
                "unit": "Wh",
                "forecast_horizon_minutes": self.artifact.forecast_horizon_minutes,
                "model_version": self.artifact.model_version,
                "status": "success",
            })

        return results


# Global singleton instance for easy import and reuse
_engine_instance: Optional[PredictionEngine] = None


def get_prediction_engine() -> PredictionEngine:
    """Returns singleton instance of PredictionEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictionEngine()
    return _engine_instance
