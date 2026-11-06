# EPred — Energy Prediction System

**EPred (Version 1.0)** is a modular, machine-learning-based energy forecasting engine designed to predict future electrical appliance energy consumption for the next 10-minute interval using historical energy measurements, indoor environmental parameters, and outdoor weather conditions.

---

## 1. Project Introduction
EPred is engineered as a reusable, testable machine learning library and prediction service. The primary objective is to build a reliable regression engine that forecasts episodic appliance energy demands without future-data leakage, providing high-precision inputs for future Home Energy Management Systems (HEMS), grid demand-response algorithms, and smart energy optimization systems.

---

## 2. Problem Statement
Household electrical appliance consumption is highly intermittent, volatile, and episodic. Spikes occur when high-power devices (washing machines, dishwashers, electric ovens) activate, followed by prolonged periods of baseline idle consumption. 

Forecasting consumption for the **next 10-minute interval** (`Appliances_Next_10min`) allows building automation systems to:
- Schedule flexible loads during low-tariff or solar-peak periods.
- Anticipate aggregate peak demand spikes.
- Detect abnormal consumption anomalies.

> **Important Distinction**:
> In this dataset, `Appliances` represents the electrical energy consumption of household appliances in Watt-hours (Wh). `lights` represents lighting fixtures energy consumption. `Appliances` does **not** represent total building electricity consumption.

---

## 3. Dataset Source and Attribution
- **Dataset**: UCI Appliances Energy Prediction
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Authors**: Luis M. Candanedo, Véronique Feldheim, Dominique Deramaix
- **Citation**:
  > Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 4. Dataset Description
The dataset contains **19,735 observations** logged at 10-minute intervals over approximately 4.5 months (from January 11, 2016 17:00:00 to May 27, 2016 18:00:00) in a low-energy house in Stambruges, Belgium.

| Variable Category | Column Names | Description |
| :--- | :--- | :--- |
| **Timestamp** | `date` | Year-month-day hour:minute:second (UTC+1) |
| **Energy Readings** | `Appliances`, `lights` | Appliance and lighting energy use in Wh |
| **Indoor Temperature** | `T1`, `T2`, `T3`, `T4`, `T5`, `T7`, `T8`, `T9` | Kitchen, Living, Laundry, Office, Bath, Ironing, Teenager, Parents (°C) |
| **Indoor Humidity** | `RH_1`, `RH_2`, `RH_3`, `RH_4`, `RH_5`, `RH_7`, `RH_8`, `RH_9` | Kitchen, Living, Laundry, Office, Bath, Ironing, Teenager, Parents (%) |
| **Outdoor Building Sensors** | `T6`, `RH_6` | Outside building north-side temperature (°C) and humidity (%) |
| **Weather Station Readings** | `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint` | Chièvres weather station environmental parameters |
| **Random Variables** | `rv1`, `rv2` | Non-predictive random variables (strictly excluded from modeling) |

---

## 5. Preprocessing Methodology
1. **Integrity Validation**: Validate all 28 original columns, timestamp continuity, duplicate records, and numeric types.
2. **Chronological Sorting**: Sort observations chronologically by `date`.
3. **Timestamp Verification**: Confirmed that 100% of consecutive readings occur at exactly 10-minute intervals (19,734 regular intervals).
4. **Outlier Investigation**: Appliance consumption contains intermittent high peaks (IQR upper bound 175 Wh, 99th percentile 576.6 Wh, maximum 1080 Wh). These observations represent genuine high-power appliance activations (e.g. laundry cycles, ovens). Deleting them would strip the model of the exact operational events it is designed to predict; hence, legitimate peaks are retained.
5. **Target Generation**:
   The forecasting target represents the appliance energy consumption in the next 10-minute interval:
   $$\text{Target} = \text{Appliances}_{t+1}$$
   Constructed in pandas via a negative single-step chronological shift:
   ```python
   df["Appliances_Next_10min"] = df["Appliances"].shift(-1)
   ```
   The final row (where shifted target is unavailable) is dropped.
6. **Warmup Trimming**: The first 12 rows are dropped because 120-minute lags (`Appliances_Lag_12`) require 12 warmup steps. Total clean samples: **19,722 rows**.
7. **Chronological Train/Validation/Test Split**:
   - Training Set (70%): Earliest 13,805 samples (Jan 11, 2016 – Apr 16, 2016)
   - Validation Set (15%): Next 2,958 samples (Apr 16, 2016 – May 07, 2016)
   - Testing Set (15%): Final 2,959 samples (May 07, 2016 – May 27, 2016)
8. **Leakage Prevention**: All scalers and transformations are fitted **strictly on the Training set**.

---

## 6. Feature Engineering Methodology
A total of **48 features** are computed deterministically:

### A. Time Features
- `Hour`, `Minute`, `Day_of_Week`, `Is_Weekend`, `Month`
- Cyclical 24-hour encoding (`Hour_Sin`, `Hour_Cos`):
  $$\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot (\text{Hour} + \text{Minute}/60)}{24}\right), \quad \text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot (\text{Hour} + \text{Minute}/60)}{24}\right)$$
- Cyclical weekly encoding (`Weekday_Sin`, `Weekday_Cos`):
  $$\text{Weekday}_{\sin} = \sin\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right), \quad \text{Weekday}_{\cos} = \cos\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$$

### B. Historical Energy Lags and Rolling Windows
Computed using only observations available at or before prediction time $t$:
- Discrete lags: `Appliances_Lag_1` (10 min prior), `Lag_2` (20 min), `Lag_3` (30 min), `Lag_6` (60 min), `Lag_12` (120 min).
- Rolling windows: `Appliances_Rolling_Mean_3`, `Rolling_Mean_6`, `Rolling_Mean_12`, `Appliances_Rolling_Std_6`.

### C. Indoor/Outdoor Environmental Gradients
- `Avg_Indoor_Temp`: Average of indoor temperature sensors (`T1`–`T5`, `T7`–`T9`).
- `Avg_Indoor_Humidity`: Average of indoor humidity sensors (`RH_1`–`RH_5`, `RH_7`–`RH_9`).
- `Temp_Difference`: `Avg_Indoor_Temp - T6`.
- `Humidity_Difference`: `Avg_Indoor_Humidity - RH_6`.

---

## 7. Algorithms Used
EPred implements and benchmarks 4 supervised regression algorithms against a persistence baseline:

1. **Persistence Baseline**: Predicts the next interval using the most recently observed appliance reading ($y_{t+1} = y_t$).
2. **Linear Regression** (`LinearRegression` with `StandardScaler`): Linear autoregressive multi-sensor mapping.
3. **Decision Tree Regressor** (`DecisionTreeRegressor(max_depth=8, min_samples_split=20)`).
4. **Random Forest Regressor** (`RandomForestRegressor(n_estimators=100, max_depth=16)`).
5. **Gradient Boosting Regressor** (`GradientBoostingRegressor(n_estimators=120, max_depth=5, lr=0.08)`).

---

## 8. Model Evaluation Results

### Validation Benchmark Comparison

| Model | Train MAE (Wh) | Train RMSE (Wh) | Train R² | Val MAE (Wh) | Val RMSE (Wh) | Val R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Persistence)** | 30.88 | 74.16 | 0.5175 | 26.14 | 66.44 | 0.4814 |
| **Linear Regression (Selected)** | 31.87 | 67.11 | 0.6049 | **27.13** | **59.71** | **0.5812** |
| **Decision Tree** | 26.80 | 57.86 | 0.7063 | 31.74 | 68.39 | 0.4505 |
| **Random Forest** | 19.01 | 41.72 | 0.8473 | 32.10 | 63.56 | 0.5254 |
| **Gradient Boosting** | 22.58 | 45.41 | 0.8191 | 43.20 | 83.40 | 0.1830 |

### Untouched Test Set Evaluation (Selected Model vs Baseline)

| Metric | Persistence Baseline | Linear Regression (EPred V1) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **MAE** | 26.76 Wh | 27.76 Wh | Comparable |
| **MSE** | 4470.19 Wh² | **3622.51 Wh²** | **-18.96%** |
| **RMSE** | 66.86 Wh | **60.19 Wh** | **-9.98%** |
| **R²** | 0.4590 | **0.5616** | **+22.35%** |

### Visual Artifacts
Generated automatically in `reports/figures/`:
- `actual_vs_predicted.png`: Scatter plot of actual vs predicted energy consumption.
- `residual_distribution.png`: Histogram and KDE distribution of forecast residuals.
- `error_over_time.png`: Temporal tracking of predicted consumption and absolute errors.
- `feature_importance.png`: Standardized coefficient magnitudes demonstrating the dominant impact of recent lags and time-of-day features.

---

## 9. Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- `pip` package manager

### Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd EPred

# 2. (Optional) Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 10. Training Instructions
To run data preprocessing, train all 4 models, compare against baseline, generate evaluation plots, and serialize the best model bundle:

```bash
python -m src.train
```

Outputs generated:
- Model bundle: `models/best_model.joblib`
- Metadata: `models/model_metadata.json`
- Comparison table: `reports/model_comparison.csv`
- Metrics: `reports/metrics.json`
- Analytical figures: `reports/figures/*.png`

---

## 11. Prediction Instructions (Python Interface)

```python
import pandas as pd
from src.predict import get_prediction_engine

# Initialize engine (loads serialized bundle once)
engine = get_prediction_engine()

# Example: Predict using raw sensor readings and 12 past intervals
sample_raw = {
    "date": "2016-05-20 14:00:00",
    "Appliances": 60.0,
    "lights": 10.0,
    "T1": 21.2, "RH_1": 42.1,
    "T2": 20.5, "RH_2": 41.0,
    "T3": 21.0, "RH_3": 40.5,
    "T4": 20.8, "RH_4": 41.2,
    "T5": 19.5, "RH_5": 50.1,
    "T6": 15.2, "RH_6": 60.5,
    "T7": 19.8, "RH_7": 38.4,
    "T8": 20.1, "RH_8": 43.2,
    "T9": 19.2, "RH_9": 41.5,
    "T_out": 14.5, "Press_mm_hg": 752.1, "RH_out": 65.0,
    "Windspeed": 4.5, "Visibility": 40.0, "Tdewpoint": 8.0,
}
# Past 12 10-minute readings [t-12 to t-1]
history = [50.0, 50.0, 60.0, 60.0, 50.0, 50.0, 60.0, 60.0, 70.0, 60.0, 60.0, 60.0]

result = engine.predict(sample_raw, raw_mode=True, historical_appliances=history)
print(result)
# Output:
# {
#   'predicted_appliances_wh': 58.42,
#   'unit': 'Wh',
#   'forecast_horizon_minutes': 10,
#   'prediction_timestamp': '2016-05-20T14:00:00',
#   'model_version': '1.0.0',
#   'model_name': 'Linear Regression',
#   'status': 'success'
# }
```

---

## 12. API Usage Instructions

### Start API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Linear Regression",
  "model_version": "1.0.0"
}
```

#### 2. Model Metadata
```bash
curl -X GET "http://localhost:8000/model/info"
```

#### 3. Single Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2016-01-11 17:00:00",
       "Appliances": 60.0,
       "lights": 20.0,
       "T1": 19.89, "RH_1": 47.59,
       "T2": 19.20, "RH_2": 44.79,
       "T3": 19.79, "RH_3": 44.73,
       "T4": 19.00, "RH_4": 45.56,
       "T5": 17.16, "RH_5": 55.20,
       "T6": 7.02,  "RH_6": 84.26,
       "T7": 17.20, "RH_7": 41.62,
       "T8": 18.20, "RH_8": 48.90,
       "T9": 17.03, "RH_9": 45.53,
       "T_out": 6.6, "Press_mm_hg": 733.5, "RH_out": 92.0,
       "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3,
       "historical_appliances": [60, 60, 50, 50, 60, 60, 50, 60, 50, 40, 50, 60]
     }'
```
Response:
```json
{
  "predicted_appliances_wh": 58.24,
  "unit": "Wh",
  "forecast_horizon_minutes": 10,
  "prediction_timestamp": "2016-01-11 17:00:00",
  "model_version": "1.0.0",
  "model_name": "Linear Regression",
  "status": "success"
}
```

#### 4. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "observations": [ ...list of observations... ]
     }'
```

---

## 13. Limitations
1. **Intermittent Spike Timing**: Household appliances are turned on by human decision at discrete intervals. While continuous auto-regression and environmental correlations capture base load and decay curves effectively, predicting the exact 10-minute minute an oven or washing machine will be switched on remains inherently probabilistic.
2. **Sub-metered Sensor Availability**: The model relies on 9 indoor temperature and humidity sensors. In homes where only a central thermostat is present, spatial indoor gradients cannot be measured without sensor imputation.
3. **Single Low-Energy Household**: Training was conducted on data from a single passive low-energy house in Belgium. Generalizing to older buildings with higher thermal inertia requires transfer learning or re-fitting.

---

## 14. Future Development Scope
- **Appliance Disaggregation (NILM)**: Integrating Non-Intrusive Load Monitoring to separate individual device power signatures.
- **Probabilistic Forecasting & Uncertainty Bounds**: Implementing quantile regression (e.g. Pinball loss) or conformal prediction to output 90% confidence bands ($[\hat{y}_{\mathrm{lower}}, \hat{y}_{\mathrm{upper}}]$) alongside point forecasts.
- **Multi-step Horizon Expansion**: Extending forecasting from 10 minutes to 1-hour, 6-hour, and 24-hour day-ahead scheduling.
- **Edge Deployment**: Compiling the pipeline into ONNX or TinyML for direct deployment onto IoT smart meters and home gateway controllers.

---

## 15. Testing
The project includes a complete test suite with 22 tests covering schema validation, feature engineering, chronological splits, leakage prevention, model loading, and API endpoints.

To run the automated tests:
```bash
pytest -v
```
