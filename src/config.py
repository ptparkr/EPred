"""Configuration settings and constants for EPred (Energy Prediction System)."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "energydata_complete.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "energydata_featured.csv"
TRAIN_DATA_PATH = DATA_DIR / "processed" / "train.csv"
VAL_DATA_PATH = DATA_DIR / "processed" / "val.csv"
TEST_DATA_PATH = DATA_DIR / "processed" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"

# Chronological Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Target Definition
TARGET_COL = "Appliances_Next_10min"
BASE_TARGET_COL = "Appliances"

# Original Raw Columns
REQUIRED_RAW_COLUMNS = [
    "date", "Appliances", "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
    "rv1", "rv2"
]

COLUMNS_TO_DROP = ["rv1", "rv2"]

INDOOR_TEMP_COLS = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_HUMIDITY_COLS = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
OUTDOOR_BUILDING_TEMP_COL = "T6"
OUTDOOR_BUILDING_HUMIDITY_COL = "RH_6"
OUTDOOR_WEATHER_TEMP_COL = "T_out"
OUTDOOR_WEATHER_HUMIDITY_COL = "RH_out"

# Feature definitions for ML
ORIGINAL_NUMERIC_FEATURES = [
    "Appliances",  # Current completed interval consumption
    "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"
]

TIME_FEATURES = [
    "Hour", "Minute", "Day_of_Week", "Is_Weekend", "Month",
    "Hour_Sin", "Hour_Cos", "Weekday_Sin", "Weekday_Cos"
]

LAG_FEATURES = [
    "Appliances_Lag_1",
    "Appliances_Lag_2",
    "Appliances_Lag_3",
    "Appliances_Lag_6",
    "Appliances_Lag_12",
    "Appliances_Rolling_Mean_3",
    "Appliances_Rolling_Mean_6",
    "Appliances_Rolling_Mean_12",
    "Appliances_Rolling_Std_6"
]

ENVIRONMENTAL_FEATURES = [
    "Avg_Indoor_Temp",
    "Avg_Indoor_Humidity",
    "Temp_Difference",
    "Humidity_Difference"
]

# All expected model input features in deterministic order
FEATURE_ORDER = (
    ORIGINAL_NUMERIC_FEATURES +
    TIME_FEATURES +
    LAG_FEATURES +
    ENVIRONMENTAL_FEATURES
)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Forecast Horizon
FORECAST_HORIZON_MINUTES = 10
MODEL_VERSION = "1.0.0"
