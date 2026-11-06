"""Model training, benchmarking, and multi-horizon calibration pipeline for EPred.

Trains baseline persistence forecasting and 4 supervised regression algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Enhancements for V2:
- Multi-horizon prediction heads (10min, 30min, 60min)
- Conformal quantile uncertainty calibration (P10, P50, P90 bounds)
- Appliance spike event classifier (predicts probability of >150 Wh surge)
- Baseline feature distribution profiling for real-time drift detection
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import (
    FEATURE_ORDER,
    TARGET_COL,
    TARGET_COL_30MIN,
    TARGET_COL_60MIN,
    BASE_TARGET_COL,
    SPIKE_THRESHOLD_WH,
    RANDOM_STATE,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    METRICS_PATH,
    COMPARISON_CSV_PATH,
)
from src.preprocessing import PreprocessingPipeline
from src.evaluate import calculate_metrics, ModelEvaluator
from src.model_registry import ModelArtifact, save_model_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_models() -> Tuple[ModelArtifact, pd.DataFrame, Dict[str, Any]]:
    """Runs complete training, model comparison, evaluation, multi-horizon calibration, and serialization."""
    # 1. Prepare data and chronological splits
    pipeline = PreprocessingPipeline()
    full_df, dataset_metadata = pipeline.prepare_dataset()
    train_df, val_df, test_df = pipeline.chronological_split(full_df)

    # Save processed sets
    pipeline.save_processed_datasets(full_df, train_df, val_df, test_df)

    # 2. Extract feature matrices and target vectors
    X_train_raw = train_df[FEATURE_ORDER]
    y_train_10m = train_df[TARGET_COL].values
    y_train_30m = train_df[TARGET_COL_30MIN].values
    y_train_60m = train_df[TARGET_COL_60MIN].values

    X_val_raw = val_df[FEATURE_ORDER]
    y_val_10m = val_df[TARGET_COL].values
    y_val_30m = val_df[TARGET_COL_30MIN].values
    y_val_60m = val_df[TARGET_COL_60MIN].values

    X_test_raw = test_df[FEATURE_ORDER]
    y_test_10m = test_df[TARGET_COL].values
    y_test_30m = test_df[TARGET_COL_30MIN].values
    y_test_60m = test_df[TARGET_COL_60MIN].values

    # 3. Fit scaler ONLY on training data to prevent leakage
    scaler = pipeline.fit_scaler(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    evaluator = ModelEvaluator()
    comparison_records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # BASELINE MODEL: Persistence (Predicts current Appliances consumption)
    # -------------------------------------------------------------
    baseline_train_pred = train_df[BASE_TARGET_COL].values
    baseline_val_pred = val_df[BASE_TARGET_COL].values
    baseline_train_metrics = calculate_metrics(y_train_10m, baseline_train_pred)
    baseline_val_metrics = calculate_metrics(y_val_10m, baseline_val_pred)

    comparison_records.append({
        "Model": "Baseline (Persistence)",
        "Train_MAE": baseline_train_metrics["MAE"],
        "Train_RMSE": baseline_train_metrics["RMSE"],
        "Train_R2": baseline_train_metrics["R2"],
        "Val_MAE": baseline_val_metrics["MAE"],
        "Val_MSE": baseline_val_metrics["MSE"],
        "Val_RMSE": baseline_val_metrics["RMSE"],
        "Val_R2": baseline_val_metrics["R2"],
    })
    logger.info(f"Baseline (Persistence) Val RMSE: {baseline_val_metrics['RMSE']}, Val R2: {baseline_val_metrics['R2']}")

    # -------------------------------------------------------------
    # REGRESSION ALGORITHMS (10-minute Horizon)
    # -------------------------------------------------------------
    candidate_models = {
        "Linear Regression": (
            LinearRegression(),
            True
        ),
        "Decision Tree": (
            DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=RANDOM_STATE
            ),
            False
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            False
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                random_state=RANDOM_STATE
            ),
            False
        )
    }

    trained_models = {}
    val_predictions = {}

    for name, (estimator, use_scaled) in candidate_models.items():
        logger.info(f"Training {name}...")
        X_tr = X_train_scaled if use_scaled else X_train_raw.values
        X_v = X_val_scaled if use_scaled else X_val_raw.values

        estimator.fit(X_tr, y_train_10m)
        y_tr_pred = estimator.predict(X_tr)
        y_v_pred = estimator.predict(X_v)

        tr_metrics = calculate_metrics(y_train_10m, y_tr_pred)
        v_metrics = calculate_metrics(y_val_10m, y_v_pred)

        trained_models[name] = (estimator, use_scaled)
        val_predictions[name] = y_v_pred

        comparison_records.append({
            "Model": name,
            "Train_MAE": tr_metrics["MAE"],
            "Train_RMSE": tr_metrics["RMSE"],
            "Train_R2": tr_metrics["R2"],
            "Val_MAE": v_metrics["MAE"],
            "Val_MSE": v_metrics["MSE"],
            "Val_RMSE": v_metrics["RMSE"],
            "Val_R2": v_metrics["R2"],
        })
        logger.info(f"{name} -> Val MAE: {v_metrics['MAE']}, Val RMSE: {v_metrics['RMSE']}, Val R2: {v_metrics['R2']}")

    # Save comparison table
    df_comparison = evaluator.save_comparison_table(comparison_records, COMPARISON_CSV_PATH)

    # 4. Select best model based on validation RMSE
    ml_candidates = [r for r in comparison_records if r["Model"] != "Baseline (Persistence)"]
    best_record = min(ml_candidates, key=lambda r: r["Val_RMSE"])
    best_model_name = best_record["Model"]
    best_estimator, best_use_scaled = trained_models[best_model_name]
    logger.info(f"Selected best model based on Val RMSE: {best_model_name} (Val RMSE: {best_record['Val_RMSE']})")

    # 5. Evaluate selected best model ONCE on the untouched test set
    X_te = X_test_scaled if best_use_scaled else X_test_raw.values
    y_test_pred = best_estimator.predict(X_te)
    test_metrics = calculate_metrics(y_test_10m, y_test_pred)
    logger.info(f"Untouched Test Set Evaluation for {best_model_name}: {test_metrics}")

    baseline_test_pred = test_df[BASE_TARGET_COL].values
    baseline_test_metrics = calculate_metrics(y_test_10m, baseline_test_pred)

    # 6. Generate analytical figures
    evaluator.plot_actual_vs_predicted(y_test_10m, y_test_pred, model_name=best_model_name)
    evaluator.plot_residual_distribution(y_test_10m, y_test_pred, model_name=best_model_name)
    evaluator.plot_prediction_error_over_time(test_df["date"], y_test_10m, y_test_pred, model_name=best_model_name)

    if hasattr(best_estimator, "feature_importances_"):
        importances = best_estimator.feature_importances_
        evaluator.plot_feature_importance(FEATURE_ORDER, importances, model_name=best_model_name)
    elif hasattr(best_estimator, "coef_"):
        importances = np.abs(best_estimator.coef_)
        evaluator.plot_feature_importance(FEATURE_ORDER, importances, model_name=f"{best_model_name} (Standardized |Coeff|)")

    # -------------------------------------------------------------
    # 7. MULTI-HORIZON HEADS (30-minute and 60-minute forecasts)
    # -------------------------------------------------------------
    logger.info("Training Multi-Horizon Forecasting Heads (30min & 60min)...")
    head_30m = Ridge(alpha=1.0)
    head_30m.fit(X_train_scaled, y_train_30m)
    val_pred_30m = head_30m.predict(X_val_scaled)
    metrics_30m = calculate_metrics(y_val_30m, val_pred_30m)
    logger.info(f"30-min Horizon Val Metrics: {metrics_30m}")

    head_60m = Ridge(alpha=1.0)
    head_60m.fit(X_train_scaled, y_train_60m)
    val_pred_60m = head_60m.predict(X_val_scaled)
    metrics_60m = calculate_metrics(y_val_60m, val_pred_60m)
    logger.info(f"60-min Horizon Val Metrics: {metrics_60m}")

    multi_horizon_estimators = {
        "30min": head_30m,
        "60min": head_60m,
    }

    # -------------------------------------------------------------
    # 8. CONFORMAL QUANTILE UNCERTAINTY CALIBRATION (P10, P50, P90)
    # -------------------------------------------------------------
    logger.info("Calibrating Conformal Quantile Offsets using validation residuals...")
    val_preds_primary = val_predictions[best_model_name]
    val_residuals = y_val_10m - val_preds_primary  # e = actual - predicted

    p10_offset = float(np.percentile(val_residuals, 10))  # Lower bound offset
    p90_offset = float(np.percentile(val_residuals, 90))  # Upper bound offset
    quantile_offsets = {
        "p10_offset": round(p10_offset, 2),
        "p90_offset": round(p90_offset, 2),
    }
    logger.info(f"Conformal Quantile Offsets: {quantile_offsets}")

    # -------------------------------------------------------------
    # 9. APPLIANCE SPIKE RISK CLASSIFIER (>150 Wh)
    # -------------------------------------------------------------
    logger.info("Training Appliance Spike Risk Classifier...")
    y_spike_train = (y_train_10m > SPIKE_THRESHOLD_WH).astype(int)
    spike_clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    spike_clf.fit(X_train_scaled, y_spike_train)

    # -------------------------------------------------------------
    # 10. FEATURE BASELINE PROFILING FOR DRIFT DETECTION
    # -------------------------------------------------------------
    feature_baselines = {}
    for col in FEATURE_ORDER:
        vals = train_df[col].values
        feature_baselines[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) or 1.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }

    # -------------------------------------------------------------
    # 11. BUNDLE AND SAVE COMPLETE MODEL ARTIFACT
    # -------------------------------------------------------------
    artifact = ModelArtifact(
        estimator=best_estimator,
        scaler=scaler,
        feature_order=FEATURE_ORDER,
        model_name=best_model_name,
        val_metrics={k: best_record[f"Val_{k}"] for k in ["MAE", "MSE", "RMSE", "R2"]},
        test_metrics=test_metrics,
        multi_horizon_estimators=multi_horizon_estimators,
        quantile_offsets=quantile_offsets,
        spike_classifier=spike_clf,
        feature_baselines=feature_baselines,
        dataset_info={
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(full_df),
            "train_date_range": [str(train_df["date"].min()), str(train_df["date"].max())],
            "val_date_range": [str(val_df["date"].min()), str(val_df["date"].max())],
            "test_date_range": [str(test_df["date"].min()), str(test_df["date"].max())],
            "baseline_val_metrics": baseline_val_metrics,
            "baseline_test_metrics": baseline_test_metrics,
            "multi_horizon_val_metrics": {
                "30min": metrics_30m,
                "60min": metrics_60m,
            }
        }
    )
    save_model_artifact(artifact, BEST_MODEL_PATH, MODEL_METADATA_PATH)

    # 12. Save structured metrics JSON
    all_metrics = {
        "selected_model": best_model_name,
        "validation_metrics": artifact.val_metrics,
        "test_metrics": artifact.test_metrics,
        "baseline_val_metrics": baseline_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "multi_horizon_val_metrics": {
            "30min": metrics_30m,
            "60min": metrics_60m,
        },
        "conformal_quantile_offsets": quantile_offsets,
        "model_comparison": comparison_records,
    }
    evaluator.save_metrics_json(all_metrics, METRICS_PATH)

    logger.info("Enhanced V2 training and calibration pipeline finished successfully.")
    return artifact, df_comparison, all_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
