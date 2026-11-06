# EPred — Model Performance and Evaluation Report (Version 2.0)

## 1. Executive Summary
EPred Version 2.0 expands the energy prediction engine from a single-step point forecast into a multi-horizon forecasting system with conformal quantile uncertainty intervals, appliance spike surge risk classification, in-memory streaming telemetry, and feature drift diagnostics.

Using chronologically structured splits (70% Train, 15% Validation, 15% Test) without data leakage, four supervised regression algorithms were benchmarked against a persistence baseline model.

## 2. Experimental Setup & Chronological Splits
- **Dataset Size**: 19,735 observations recorded every 10 minutes between January 11, 2016 and May 27, 2016 (4.5 months) in Stambruges, Belgium.
- **Preprocessing**: 12 warmup lag rows and 6 terminal multi-horizon target rows were dropped, yielding 19,717 modeling rows.
- **Split Breakdown**:
  - **Training Set (70%)**: 13,801 samples (2016-01-11 19:00:00 to 2016-04-16 15:00:00)
  - **Validation Set (15%)**: 2,958 samples (2016-04-16 15:10:00 to 2016-05-07 04:00:00)
  - **Testing Set (15%)**: 2,958 samples (2016-05-07 04:10:00 to 2016-05-27 17:00:00)

## 3. Benchmark and Model Comparison Results (10-Minute Horizon)

| Model | Train MAE (Wh) | Train RMSE (Wh) | Train R² | Val MAE (Wh) | Val RMSE (Wh) | Val R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Persistence)** | 30.88 | 74.16 | 0.5175 | 26.14 | 66.45 | 0.4813 |
| **Linear Regression (Selected)** | 31.87 | 67.11 | 0.6049 | **27.15** | **59.73** | **0.5810** |
| **Decision Tree** | 26.80 | 57.86 | 0.7063 | 31.74 | 68.40 | 0.4504 |
| **Random Forest** | 19.01 | 41.72 | 0.8473 | 32.16 | 63.44 | 0.5272 |
| **Gradient Boosting** | 22.58 | 45.41 | 0.8191 | 55.71 | 103.07 | -0.2480 |

## 4. Multi-Horizon Forecasting Performance

| Forecast Horizon | Forward Steps | Val MAE (Wh) | Val RMSE (Wh) | Val R² |
| :--- | :---: | :---: | :---: | :---: |
| **10-minute horizon** ($t+1$) | 1 step | 27.15 | 59.73 | 0.5810 |
| **30-minute horizon** ($t+3$) | 3 steps | 40.85 | 81.21 | 0.2252 |
| **60-minute horizon** ($t+6$) | 6 steps | 45.21 | 84.05 | 0.1696 |

## 5. Conformal Quantile Calibration
Using the empirical residuals on the held-out validation set ($e = y_{\mathrm{val}} - \hat{y}_{\mathrm{val}}$), distribution-free conformal quantile offsets were calibrated:
- **Lower bound ($P_{10}$)**: $-32.26\text{ Wh}$ offset
- **Upper bound ($P_{90}$)**: $+20.48\text{ Wh}$ offset
- Resulting interval: $[P_{10}, P_{50}, P_{90}] = [\max(0, \hat{y} - 32.26), \; \hat{y}, \; \hat{y} + 20.48]$.

## 6. Untouched Test Set Evaluation
The winning model (**Linear Regression**) was evaluated strictly ONCE on the held-out unseen Test set:

| Evaluation Metric | Baseline (Persistence) | Linear Regression (EPred V2) | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | 26.76 Wh | 27.64 Wh | Comparable |
| **MSE (Mean Squared Error)** | 4,470.19 Wh² | 3,599.19 Wh² | **-19.48%** |
| **RMSE (Root Mean Squared Error)** | 66.86 Wh | **59.99 Wh** | **-10.28%** |
| **R² (Coefficient of Determination)** | 0.4590 | **0.5603** | **+22.07%** |
