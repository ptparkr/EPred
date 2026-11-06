# EPred — Model Performance and Evaluation Report

## 1. Executive Summary
EPred (Energy Prediction System) Version 1.0 forecasts household appliance electrical energy consumption for the upcoming 10-minute interval (`Appliances_Next_10min`) using the real-world UCI Appliances Energy Prediction dataset. 

Using chronologically structured splits (70% Train, 15% Validation, 15% Test) without data leakage, four supervised regression algorithms were evaluated and compared against a persistence baseline model.

## 2. Experimental Setup & Chronological Splits
- **Dataset Size**: 19,735 observations recorded every 10 minutes between January 11, 2016 and May 27, 2016 (4.5 months) in Stambruges, Belgium.
- **Preprocessing**: 12 warmup lag rows and 1 terminal target row were dropped, yielding 19,722 modeling rows.
- **Split Breakdown**:
  - **Training Set (70%)**: 13,805 samples (2016-01-11 19:00:00 to 2016-04-16 15:40:00)
  - **Validation Set (15%)**: 2,958 samples (2016-04-16 15:50:00 to 2016-05-07 04:40:00)
  - **Testing Set (15%)**: 2,959 samples (2016-05-07 04:50:00 to 2016-05-27 17:50:00)
- **Features (48 Total)**:
  - Original room environmental sensors: Kitchen, Living Room, Laundry, Office, Bathroom, Ironing, Teenager, Parents (T1–T9, RH_1–RH_9).
  - Outdoor weather station readings: T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint.
  - Energy readings: Latest completed `Appliances` and `lights`.
  - Engineered time features: Hour, Minute, Day_of_Week, Is_Weekend, Month, cyclical Hour_Sin/Cos, cyclical Weekday_Sin/Cos.
  - Engineered lag/rolling features: Appliances_Lag_1, Lag_2, Lag_3, Lag_6, Lag_12, Rolling_Mean_3, Rolling_Mean_6, Rolling_Mean_12, Rolling_Std_6.
  - Engineered environmental gradients: Avg_Indoor_Temp, Avg_Indoor_Humidity, Temp_Difference, Humidity_Difference.

## 3. Benchmark and Model Comparison Results

| Model | Train MAE (Wh) | Train RMSE (Wh) | Train R² | Val MAE (Wh) | Val RMSE (Wh) | Val R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Persistence)** | 30.88 | 74.16 | 0.5175 | 26.14 | 66.44 | 0.4814 |
| **Linear Regression** | 31.87 | 67.11 | 0.6049 | **27.13** | **59.71** | **0.5812** |
| **Decision Tree** | 26.80 | 57.86 | 0.7063 | 31.74 | 68.39 | 0.4505 |
| **Random Forest** | 19.01 | 41.72 | 0.8473 | 32.10 | 63.56 | 0.5254 |
| **Gradient Boosting** | 22.58 | 45.41 | 0.8191 | 43.20 | 83.40 | 0.1830 |

### Key Observations:
1. **Model Selection**: Linear Regression achieved the lowest Validation RMSE (59.71 Wh) and highest Validation R² (0.5812), significantly outperforming the persistence baseline (66.44 Wh RMSE).
2. **Overfitting in Non-Linear Trees**: Decision Trees, Random Forests, and Gradient Boosting displayed signs of overfitting to training regimes (Winter conditions), struggling to generalize when weather and humidity transitioned from Winter into Spring on the validation set.
3. **Linear Generalization**: Standardized linear relationships between recent energy lags and time cycle features generalize reliably across seasonal changes.

## 4. Untouched Test Set Evaluation
The winning model (**Linear Regression**) was evaluated strictly ONCE on the held-out unseen Test set:

| Evaluation Metric | Baseline (Persistence) | Linear Regression (EPred V1) | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | 26.76 Wh | 27.76 Wh | Comparable |
| **MSE (Mean Squared Error)** | 4470.19 Wh² | 3622.51 Wh² | **-18.96%** |
| **RMSE (Root Mean Squared Error)** | 66.86 Wh | **60.19 Wh** | **-9.98%** |
| **R² (Coefficient of Determination)** | 0.4590 | **0.5616** | **+22.35%** |

## 5. Statistical and Metric Interpretation
- **Significance of MAE vs RMSE**:
  - MAE (27.76 Wh) measures the expected magnitude of typical forecast errors across quiet base-load periods (~40–60 Wh).
  - RMSE (60.19 Wh) penalizes larger errors quadratically. Because household appliance activations are intermittent and episodic (spiking to 800+ Wh when ovens, heaters, or washing machines switch on), large RMSE values are driven by sudden appliance switches. A lower RMSE indicates substantially better dampening of large spike errors.
- **Limitations of R² in Episodic Time Series**:
  - $R^2$ indicates the proportion of variance explained ($56.16\%$). While lower than synthetic benchmarks, an $R^2 > 0.56$ on high-frequency 10-minute intermittent residential loads without future knowledge represents a strong operational forecast.
