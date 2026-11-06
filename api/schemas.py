"""Pydantic schemas and data validation models for EPred V2 API."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_name: str = Field(..., examples=["Linear Regression"])
    model_version: str = Field(..., examples=["2.0.0"])


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., examples=["Linear Regression"])
    model_version: str = Field(..., examples=["2.0.0"])
    forecast_horizon_minutes: int = Field(..., examples=[10])
    supported_horizons_minutes: List[int] = Field(..., examples=[[10, 30, 60]])
    training_timestamp: str
    num_features: int = Field(..., examples=[48])
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    conformal_offsets: Dict[str, float]
    has_spike_classifier: bool
    ring_buffer_size: int
    ring_buffer_warmed_up: bool


class ConfidenceIntervals(BaseModel):
    p10_wh: float = Field(..., examples=[42.5], description="10th percentile conservative lower bound (Wh)")
    p50_median_wh: float = Field(..., examples=[58.2], description="50th percentile expected median consumption (Wh)")
    p90_wh: float = Field(..., examples=[115.0], description="90th percentile upper risk bound (Wh)")


class SpikeRisk(BaseModel):
    probability: float = Field(..., examples=[0.145], description="Estimated probability of appliance load exceeding 150 Wh")
    risk_level: str = Field(..., examples=["low"], description="Risk tier: 'low' (<20%), 'medium' (20-50%), 'high' (>50%)")
    spike_threshold_wh: float = Field(150.0, examples=[150.0])


class RawSensorInput(BaseModel):
    """Room-specific environmental and weather measurements with client-supplied history."""
    date: Optional[str] = Field(None, examples=["2016-01-11 17:00:00"], description="Observation timestamp")
    Appliances: float = Field(..., ge=0, examples=[60.0], description="Latest completed appliance consumption reading in Wh")
    lights: float = Field(..., ge=0, examples=[20.0], description="Lighting consumption in Wh")
    
    # Kitchen (Area 1)
    T1: float = Field(..., examples=[19.89], description="Kitchen temperature in Celsius")
    RH_1: float = Field(..., ge=0, le=100, examples=[47.59], description="Kitchen humidity in %")
    
    # Living Room (Area 2)
    T2: float = Field(..., examples=[19.2], description="Living room temperature in Celsius")
    RH_2: float = Field(..., ge=0, le=100, examples=[44.79], description="Living room humidity in %")
    
    # Laundry Room (Area 3)
    T3: float = Field(..., examples=[19.79], description="Laundry room temperature in Celsius")
    RH_3: float = Field(..., ge=0, le=100, examples=[44.73], description="Laundry room humidity in %")
    
    # Office Room (Area 4)
    T4: float = Field(..., examples=[19.0], description="Office temperature in Celsius")
    RH_4: float = Field(..., ge=0, le=100, examples=[45.56], description="Office humidity in %")
    
    # Bathroom (Area 5)
    T5: float = Field(..., examples=[17.16], description="Bathroom temperature in Celsius")
    RH_5: float = Field(..., ge=0, le=100, examples=[55.2], description="Bathroom humidity in %")
    
    # Outside North Side Building (Area 6)
    T6: float = Field(..., examples=[7.02], description="Outside north building temperature in Celsius")
    RH_6: float = Field(..., ge=0, le=100, examples=[84.26], description="Outside north building humidity in %")
    
    # Ironing Room (Area 7)
    T7: float = Field(..., examples=[17.2], description="Ironing room temperature in Celsius")
    RH_7: float = Field(..., ge=0, le=100, examples=[41.62], description="Ironing room humidity in %")
    
    # Teenager Room (Area 8)
    T8: float = Field(..., examples=[18.2], description="Teenager room temperature in Celsius")
    RH_8: float = Field(..., ge=0, le=100, examples=[48.9], description="Teenager room humidity in %")
    
    # Parents Room (Area 9)
    T9: float = Field(..., examples=[17.03], description="Parents room temperature in Celsius")
    RH_9: float = Field(..., ge=0, le=100, examples=[45.53], description="Parents room humidity in %")
    
    # Weather Station Readings
    T_out: float = Field(..., examples=[6.6], description="Outside temperature from weather station in Celsius")
    Press_mm_hg: float = Field(..., examples=[733.5], description="Atmospheric pressure in mm Hg")
    RH_out: float = Field(..., ge=0, le=100, examples=[92.0], description="Outside humidity from weather station in %")
    Windspeed: float = Field(..., ge=0, examples=[7.0], description="Windspeed in m/s")
    Visibility: float = Field(..., ge=0, examples=[63.0], description="Visibility in km")
    Tdewpoint: float = Field(..., examples=[5.3], description="Dew point temperature in Celsius")

    # Historical consumption for lag/rolling feature generation
    historical_appliances: List[float] = Field(
        ...,
        description="Historical appliance consumption readings for preceding 10-minute intervals (at least 12 readings in chronological order [t-12 to t-1])",
        examples=[[60.0, 60.0, 50.0, 50.0, 60.0, 60.0, 50.0, 60.0, 50.0, 40.0, 50.0, 60.0]]
    )

    @field_validator("historical_appliances")
    @classmethod
    def validate_historical_len(cls, v: List[float]) -> List[float]:
        if len(v) < 12:
            raise ValueError(f"historical_appliances must contain at least 12 values, got {len(v)}.")
        for idx, val in enumerate(v):
            if val < 0:
                raise ValueError(f"Historical reading at index {idx} is negative ({val}). Consumption must be non-negative.")
        return v


class TelemetryStreamInput(BaseModel):
    """Real-time streaming telemetry input. Requires NO historical payload array."""
    date: Optional[str] = Field(None, examples=["2016-01-11 17:00:00"])
    Appliances: float = Field(..., ge=0, examples=[60.0])
    lights: float = Field(0.0, ge=0, examples=[10.0])
    T1: float = Field(..., examples=[20.0])
    RH_1: float = Field(..., ge=0, le=100, examples=[45.0])
    T2: float = Field(..., examples=[19.5])
    RH_2: float = Field(..., ge=0, le=100, examples=[44.0])
    T3: float = Field(..., examples=[20.2])
    RH_3: float = Field(..., ge=0, le=100, examples=[43.5])
    T4: float = Field(..., examples=[19.8])
    RH_4: float = Field(..., ge=0, le=100, examples=[42.0])
    T5: float = Field(..., examples=[18.5])
    RH_5: float = Field(..., ge=0, le=100, examples=[50.0])
    T6: float = Field(..., examples=[8.5])
    RH_6: float = Field(..., ge=0, le=100, examples=[80.0])
    T7: float = Field(..., examples=[18.0])
    RH_7: float = Field(..., ge=0, le=100, examples=[40.0])
    T8: float = Field(..., examples=[19.0])
    RH_8: float = Field(..., ge=0, le=100, examples=[45.0])
    T9: float = Field(..., examples=[18.2])
    RH_9: float = Field(..., ge=0, le=100, examples=[43.0])
    T_out: float = Field(..., examples=[7.0])
    Press_mm_hg: float = Field(..., examples=[740.0])
    RH_out: float = Field(..., ge=0, le=100, examples=[85.0])
    Windspeed: float = Field(..., ge=0, examples=[5.0])
    Visibility: float = Field(..., ge=0, examples=[50.0])
    Tdewpoint: float = Field(..., examples=[4.5])


class PredictionResponse(BaseModel):
    predicted_appliances_wh: float = Field(..., examples=[58.24], description="Predicted consumption for next 10-minute interval in Wh")
    unit: str = Field("Wh", examples=["Wh"])
    forecast_horizon_minutes: int = Field(10, examples=[10])
    prediction_timestamp: str = Field(..., examples=["2016-01-11T17:00:00"])
    model_version: str = Field(..., examples=["2.0.0"])
    model_name: str = Field(..., examples=["Linear Regression"])
    confidence_intervals: Optional[ConfidenceIntervals] = None
    spike_risk: Optional[SpikeRisk] = None
    multi_horizon_forecast: Optional[Dict[str, float]] = None
    status: str = Field("success", examples=["success"])


class BatchPredictionRequest(BaseModel):
    observations: List[RawSensorInput] = Field(
        ...,
        description="List of raw sensor observations with historical readings",
        min_length=1
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    count: int
    status: str = Field("success", examples=["success"])


class DriftReportResponse(BaseModel):
    status: str = Field(..., examples=["normal"])
    drifted_features_count: int = Field(..., examples=[0])
    total_features_checked: int = Field(..., examples=[48])
    details: Dict[str, Any]
