"""Model training and benchmarking pipeline for EPred.

Implements baseline persistence forecasting and 4 supervised regression algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Performs chronological validation, selects best model based on validation RMSE,
evaluates once on untouched test set, generates plots, and persists the best model artifact.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import (
    FEATURE_ORDER,
    TARGET_COL,
    BASE_TARGET_COL,
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
    """Runs complete training, model comparison, evaluation, and serialization pipeline."""
    # 1. Prepare data and chronological splits
    pipeline = PreprocessingPipeline()
    full_df, dataset_metadata = pipeline.prepare_dataset()
    train_df, val_df, test_df = pipeline.chronological_split(full_df)

    # Save processed sets
    pipeline.save_processed_datasets(full_df, train_df, val_df, test_df)

    # 2. Extract feature matrices and target vectors
    X_train_raw = train_df[FEATURE_ORDER]
    y_train = train_df[TARGET_COL].values

    X_val_raw = val_df[FEATURE_ORDER]
    y_val = val_df[TARGET_COL].values

    X_test_raw = test_df[FEATURE_ORDER]
    y_test = test_df[TARGET_COL].values

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
    baseline_train_metrics = calculate_metrics(y_train, baseline_train_pred)
    baseline_val_metrics = calculate_metrics(y_val, baseline_val_pred)

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
    # REGRESSION ALGORITHMS
    # -------------------------------------------------------------
    candidate_models = {
        "Linear Regression": (
            LinearRegression(),
            True  # Use scaled features
        ),
        "Decision Tree": (
            DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=RANDOM_STATE
            ),
            False  # Tree models can use raw or scaled
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

        estimator.fit(X_tr, y_train)
        y_tr_pred = estimator.predict(X_tr)
        y_v_pred = estimator.predict(X_v)

        tr_metrics = calculate_metrics(y_train, y_tr_pred)
        v_metrics = calculate_metrics(y_val, y_v_pred)

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
    # Exclude baseline from selection candidates
    ml_candidates = [r for r in comparison_records if r["Model"] != "Baseline (Persistence)"]
    best_record = min(ml_candidates, key=lambda r: r["Val_RMSE"])
    best_model_name = best_record["Model"]
    best_estimator, best_use_scaled = trained_models[best_model_name]
    logger.info(f"Selected best model based on Val RMSE: {best_model_name} (Val RMSE: {best_record['Val_RMSE']})")

    # 5. Evaluate selected best model ONCE on the untouched test set
    X_te = X_test_scaled if best_use_scaled else X_test_raw.values
    y_test_pred = best_estimator.predict(X_te)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    logger.info(f"Untouched Test Set Evaluation for {best_model_name}: {test_metrics}")

    # Baseline test metrics for benchmark comparison
    baseline_test_pred = test_df[BASE_TARGET_COL].values
    baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred)
    logger.info(f"Untouched Test Set Baseline Evaluation: {baseline_test_metrics}")

    # 6. Generate analytical figures
    evaluator.plot_actual_vs_predicted(
        y_test, y_test_pred, model_name=best_model_name
    )
    evaluator.plot_residual_distribution(
        y_test, y_test_pred, model_name=best_model_name
    )
    evaluator.plot_prediction_error_over_time(
        test_df["date"], y_test, y_test_pred, model_name=best_model_name
    )

    # Feature importances / standardized coefficients
    if hasattr(best_estimator, "feature_importances_"):
        importances = best_estimator.feature_importances_
        evaluator.plot_feature_importance(
            FEATURE_ORDER, importances, model_name=best_model_name
        )
    elif hasattr(best_estimator, "coef_"):
        # For linear models with standardized features, abs(coef_) reflects feature importance
        importances = np.abs(best_estimator.coef_)
        evaluator.plot_feature_importance(
            FEATURE_ORDER, importances, model_name=f"{best_model_name} (Standardized |Coeff|)"
        )
            FEATURE_ORDER, importances, model_name=best_model_name
        )

    # 7. Construct and save ModelArtifact bundle
    artifact = ModelArtifact(
        estimator=best_estimator,
        scaler=scaler,
        feature_order=FEATURE_ORDER,
        model_name=best_model_name,
        val_metrics={k: best_record[f"Val_{k}"] for k in ["MAE", "MSE", "RMSE", "R2"]},
        test_metrics=test_metrics,
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
        }
    )
    save_model_artifact(artifact, BEST_MODEL_PATH, MODEL_METADATA_PATH)

    # 8. Save structured metrics JSON
    all_metrics = {
        "selected_model": best_model_name,
        "validation_metrics": artifact.val_metrics,
        "test_metrics": artifact.test_metrics,
        "baseline_val_metrics": baseline_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "model_comparison": comparison_records,
    }
    evaluator.save_metrics_json(all_metrics, METRICS_PATH)

    logger.info("Training and evaluation pipeline finished successfully.")
    return artifact, df_comparison, all_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
