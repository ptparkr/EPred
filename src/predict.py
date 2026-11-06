"""Prediction engine module for EPred.

Provides a unified, validated interface for single, batch, and streaming predictions.
Supports:
- Point regression and multi-horizon trajectory forecasting (10m, 30m, 60m)
- Conformal quantile uncertainty intervals (P10, P50, P90)
- Appliance spike surge risk classification
- In-memory telemetry ring buffer for zero-client-history streaming
- Real-time input drift diagnostics against training baselines
"""

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_ORDER,
    FORECAST_HORIZON_MINUTES,
    SUPPORTED_HORIZONS_MINUTES,
    MODEL_VERSION,
    TELEMETRY_BUFFER_SIZE,
    INDOOR_TEMP_COLS,
    INDOOR_HUMIDITY_COLS,
    OUTDOOR_BUILDING_TEMP_COL,
    OUTDOOR_BUILDING_HUMIDITY_COL,
)
from src.model_registry import ModelArtifact, load_model_artifact

logger = logging.getLogger(__name__)


class TelemetryRingBuffer:
    """Thread-safe circular ring buffer for caching historical appliance readings."""

    def __init__(self, maxlen: int = TELEMETRY_BUFFER_SIZE):
        self.buffer = deque(maxlen=maxlen)

    def append(self, appliance_reading: float) -> None:
        """Appends a new verified completed appliance reading to the buffer."""
        self.buffer.append(float(appliance_reading))

    def is_warmed_up(self) -> bool:
        """Returns True if the buffer has accumulated at least maxlen readings."""
        return len(self.buffer) == self.buffer.maxlen

    def get_history(self) -> List[float]:
        """Returns ordered list of historical readings."""
        return list(self.buffer)

    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class PredictionEngine:
    """Production inference and diagnostic engine for EPred."""

    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        self.artifact: Optional[ModelArtifact] = None
        self.ring_buffer = TelemetryRingBuffer(maxlen=TELEMETRY_BUFFER_SIZE)
        self._load_model()

    def _load_model(self) -> None:
        """Loads and verifies model artifact from storage."""
        self.artifact = load_model_artifact(self.model_path)
        self.expected_features = self.artifact.feature_order

    def is_healthy(self) -> bool:
        """Returns True if the engine is initialized and ready for inference."""
        return self.artifact is not None and self.artifact.estimator is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active inference model and multi-horizon capabilities."""
        if not self.is_healthy():
            raise RuntimeError("Prediction engine model is not loaded.")
        return {
            "model_name": self.artifact.model_name,
            "model_version": self.artifact.model_version,
            "forecast_horizon_minutes": self.artifact.forecast_horizon_minutes,
            "supported_horizons_minutes": SUPPORTED_HORIZONS_MINUTES,
            "training_timestamp": self.artifact.training_timestamp,
            "num_features": len(self.expected_features),
            "validation_metrics": self.artifact.val_metrics,
            "test_metrics": self.artifact.test_metrics,
            "conformal_offsets": self.artifact.quantile_offsets,
            "has_spike_classifier": self.artifact.spike_classifier is not None,
            "ring_buffer_size": self.ring_buffer.size(),
            "ring_buffer_warmed_up": self.ring_buffer.is_warmed_up(),
        }

    def compute_features_from_raw(
        self,
        raw_reading: Dict[str, Any],
        historical_appliances: List[float],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, float]:
        """Derives all required features from raw sensor readings and historical consumption."""
        if not historical_appliances or len(historical_appliances) < 12:
            raise ValueError(
                f"Insufficient historical data: received {len(historical_appliances) if historical_appliances else 0} "
                "historical readings. Exactly or at least 12 past 10-minute readings are required to compute lag and rolling features."
            )

        history_arr = np.asarray(historical_appliances, dtype=float)
        if not np.all(np.isfinite(history_arr)):
            raise ValueError("Historical appliances series contains NaN or infinite values.")

        curr_appliances = float(raw_reading.get("Appliances", 0))
        if not np.isfinite(curr_appliances) or curr_appliances < 0:
            raise ValueError(f"Invalid current Appliances value: {curr_appliances}. Must be non-negative finite number.")

        lag_1 = float(history_arr[-1])
        lag_2 = float(history_arr[-2])
        lag_3 = float(history_arr[-3])
        lag_6 = float(history_arr[-6])
        lag_12 = float(history_arr[-12])

        window_3 = np.append(history_arr[-2:], curr_appliances)
        window_6 = np.append(history_arr[-5:], curr_appliances)
        window_12 = np.append(history_arr[-11:], curr_appliances)

        roll_mean_3 = float(np.mean(window_3))
        roll_mean_6 = float(np.mean(window_6))
        roll_mean_12 = float(np.mean(window_12))
        roll_std_6 = float(np.std(window_6, ddof=1)) if len(window_6) > 1 else 0.0

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
        """Generates comprehensive single-step and multi-horizon forecasts with uncertainty bounds."""
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

        if self.artifact.scaler is not None:
            X_scaled = self.artifact.scaler.transform(ordered_df)
        else:
            X_scaled = ordered_df.values

        # 1. Primary 10-minute prediction
        raw_pred_10m = float(self.artifact.estimator.predict(X_scaled)[0])
        predicted_wh = max(0.0, round(raw_pred_10m, 2))

        # 2. Conformal Quantile Bounds (P10, P50, P90)
        p10_off = self.artifact.quantile_offsets.get("p10_offset", -20.0)
        p90_off = self.artifact.quantile_offsets.get("p90_offset", 45.0)
        p10 = max(0.0, round(predicted_wh + p10_off, 2))
        p50 = predicted_wh
        p90 = max(0.0, round(predicted_wh + p90_off, 2))

        confidence_intervals = {
            "p10_wh": p10,
            "p50_median_wh": p50,
            "p90_wh": p90,
        }

        # 3. Appliance Spike Risk Assessment
        spike_prob = 0.0
        spike_level = "low"
        if self.artifact.spike_classifier is not None:
            spike_prob = float(self.artifact.spike_classifier.predict_proba(X_scaled)[0][1])
            spike_prob = round(spike_prob, 3)
            if spike_prob >= 0.50:
                spike_level = "high"
            elif spike_prob >= 0.20:
                spike_level = "medium"

        # 4. Multi-Horizon Forecasting Trajectory (10m, 30m, 60m)
        multi_horizon = {
            "10min": predicted_wh,
        }
        for horizon_key, head_estimator in self.artifact.multi_horizon_estimators.items():
            head_val = float(head_estimator.predict(X_scaled)[0])
            multi_horizon[horizon_key] = max(0.0, round(head_val, 2))

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
            "confidence_intervals": confidence_intervals,
            "spike_risk": {
                "probability": spike_prob,
                "risk_level": spike_level,
                "spike_threshold_wh": 150.0,
            },
            "multi_horizon_forecast": multi_horizon,
            "status": "success",
        }

    def predict_telemetry_stream(
        self,
        telemetry: Dict[str, Any],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Accepts a single real-time telemetry observation using internal ring buffer.

        Clients do not need to provide historical array payloads!
        """
        curr_app = float(telemetry.get("Appliances", 0))
        # Check if buffer has enough historical data
        if not self.ring_buffer.is_warmed_up():
            self.ring_buffer.append(curr_app)
            return {
                "status": "warming_up",
                "message": (
                    f"Telemetry buffer warming up ({self.ring_buffer.size()}/{TELEMETRY_BUFFER_SIZE} readings). "
                    f"Need {TELEMETRY_BUFFER_SIZE - self.ring_buffer.size()} more intervals to begin predictions."
                ),
                "buffer_size": self.ring_buffer.size(),
                "required_size": TELEMETRY_BUFFER_SIZE,
            }

        history = self.ring_buffer.get_history()
        # Predict using current history
        result = self.predict(
            features=telemetry,
            raw_mode=True,
            historical_appliances=history,
            timestamp=timestamp,
        )
        # Advance ring buffer with current reading
        self.ring_buffer.append(curr_app)
        return result

    def check_drift(self, incoming_features: pd.DataFrame) -> Dict[str, Any]:
        """Calculates drift diagnostics against training feature baselines."""
        if not self.artifact.feature_baselines:
            return {"status": "no_baseline_available"}

        drift_details = {}
        drift_count = 0

        for col in self.expected_features:
            if col not in incoming_features.columns:
                continue
            base = self.artifact.feature_baselines[col]
            curr_val = float(incoming_features[col].mean())
            z_score = abs(curr_val - base["mean"]) / base["std"]

            is_drifted = bool(z_score > 3.0)
            if is_drifted:
                drift_count += 1

            drift_details[col] = {
                "current_mean": round(curr_val, 3),
                "baseline_mean": round(base["mean"], 3),
                "baseline_std": round(base["std"], 3),
                "z_score": round(z_score, 2),
                "drift_flag": is_drifted,
            }

        status = "normal"
        if drift_count > 5:
            status = "critical_drift"
        elif drift_count > 0:
            status = "minor_drift"

        return {
            "status": status,
            "drifted_features_count": drift_count,
            "total_features_checked": len(drift_details),
            "details": drift_details,
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
        p10_off = self.artifact.quantile_offsets.get("p10_offset", -20.0)
        p90_off = self.artifact.quantile_offsets.get("p90_offset", 45.0)

        results = []
        for i, val in enumerate(raw_preds):
            predicted_wh = max(0.0, round(float(val), 2))
            results.append({
                "index": i,
                "predicted_appliances_wh": predicted_wh,
                "unit": "Wh",
                "forecast_horizon_minutes": self.artifact.forecast_horizon_minutes,
                "confidence_intervals": {
                    "p10_wh": max(0.0, round(predicted_wh + p10_off, 2)),
                    "p50_median_wh": predicted_wh,
                    "p90_wh": max(0.0, round(predicted_wh + p90_off, 2)),
                },
                "model_version": self.artifact.model_version,
                "status": "success",
            })

        return results


# Global singleton instance
_engine_instance: Optional[PredictionEngine] = None


def get_prediction_engine() -> PredictionEngine:
    """Returns singleton instance of PredictionEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictionEngine()
    return _engine_instance
