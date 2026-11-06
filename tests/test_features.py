"""Unit tests for EPred feature engineering and zero data-leakage enforcement."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    generate_time_features,
    generate_lag_features,
    generate_environmental_features,
    generate_all_features,
)


@pytest.fixture
def sample_timeseries_df():
    """Generates a controlled 20-step synthetic sequence with known values to verify math."""
    dates = pd.date_range("2016-01-01 00:00:00", periods=20, freq="10min")
    appliances = np.arange(10, 210, 10, dtype=float)  # 10, 20, 30, ..., 200
    df = pd.DataFrame({
        "date": dates,
        "Appliances": appliances,
        "lights": 10.0,
        "T1": 20.0, "T2": 20.0, "T3": 20.0, "T4": 20.0, "T5": 20.0,
        "T7": 20.0, "T8": 20.0, "T9": 20.0,
        "RH_1": 50.0, "RH_2": 50.0, "RH_3": 50.0, "RH_4": 50.0, "RH_5": 50.0,
        "RH_7": 50.0, "RH_8": 50.0, "RH_9": 50.0,
        "T6": 5.0, "RH_6": 80.0,
        "T_out": 4.0, "Press_mm_hg": 750.0, "RH_out": 85.0,
        "Windspeed": 3.0, "Visibility": 40.0, "Tdewpoint": 2.0
    })
    return df


def test_time_features_cyclical_bounds(sample_timeseries_df):
    df_feat = generate_time_features(sample_timeseries_df)

    assert "Hour" in df_feat.columns
    assert "Minute" in df_feat.columns
    assert "Hour_Sin" in df_feat.columns
    assert "Hour_Cos" in df_feat.columns
    assert "Weekday_Sin" in df_feat.columns
    assert "Weekday_Cos" in df_feat.columns

    # Cyclical sin and cos must strictly remain within [-1, 1]
    assert df_feat["Hour_Sin"].min() >= -1.0
    assert df_feat["Hour_Sin"].max() <= 1.0
    assert df_feat["Hour_Cos"].min() >= -1.0
    assert df_feat["Hour_Cos"].max() <= 1.0

    # Test pythagorean identity sin^2 + cos^2 = 1
    trig_sum = df_feat["Hour_Sin"] ** 2 + df_feat["Hour_Cos"] ** 2
    assert np.allclose(trig_sum, 1.0, atol=1e-5)


def test_lag_features_math_and_no_future_leakage(sample_timeseries_df):
    df_feat = generate_lag_features(sample_timeseries_df)

    # For index 1 (Appliances=20), Lag_1 must be 10 (index 0)
    assert df_feat["Appliances_Lag_1"].iloc[1] == 10.0
    # For index 2 (Appliances=30), Lag_1 is 20, Lag_2 is 10
    assert df_feat["Appliances_Lag_1"].iloc[2] == 20.0
    assert df_feat["Appliances_Lag_2"].iloc[2] == 10.0

    # Verify rolling mean over 3 intervals (index 2: [10, 20, 30] -> mean = 20)
    assert np.isclose(df_feat["Appliances_Rolling_Mean_3"].iloc[2], 20.0)

    # CRITICAL LEAKAGE TEST:
    # Modifying future rows (index > t) MUST NOT alter features at row t
    sample_df_copy = sample_timeseries_df.copy()
    sample_df_copy.loc[15:, "Appliances"] = 9999.0

    df_feat_modified = generate_lag_features(sample_df_copy)
    # Rows 0 through 14 should be completely identical
    pd.testing.assert_series_equal(
        df_feat["Appliances_Lag_1"].iloc[:15],
        df_feat_modified["Appliances_Lag_1"].iloc[:15]
    )
    pd.testing.assert_series_equal(
        df_feat["Appliances_Rolling_Mean_3"].iloc[:15],
        df_feat_modified["Appliances_Rolling_Mean_3"].iloc[:15]
    )


def test_environmental_features_math(sample_timeseries_df):
    df_feat = generate_environmental_features(sample_timeseries_df)

    # All 8 indoor temps are 20.0, so Avg_Indoor_Temp must be 20.0
    assert np.allclose(df_feat["Avg_Indoor_Temp"], 20.0)
    # All 8 indoor humidities are 50.0, so Avg_Indoor_Humidity must be 50.0
    assert np.allclose(df_feat["Avg_Indoor_Humidity"], 50.0)

    # Temp_Difference = 20.0 - 5.0 (T6) = 15.0
    assert np.allclose(df_feat["Temp_Difference"], 15.0)
    # Humidity_Difference = 50.0 - 80.0 (RH_6) = -30.0
    assert np.allclose(df_feat["Humidity_Difference"], -30.0)
