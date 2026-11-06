"""Model evaluation module for EPred.

Computes MAE, MSE, RMSE, and R2 evaluation metrics, generates comparison tables,
and creates visualization figures (actual vs predicted, residuals, error over time, feature importance).
"""

import os
import tempfile
# Ensure Matplotlib config uses writable temp dir
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_PATH, COMPARISON_CSV_PATH

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates regression metrics: MAE, MSE, RMSE, and R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


class ModelEvaluator:
    """Handles comprehensive evaluation, comparison reporting, and visualization."""

    def __init__(self, figures_dir: Path = FIGURES_DIR):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clean styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["figure.dpi"] = 150

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
        sample_size: int = 500
    ) -> Path:
        """Plots scatter plot of actual vs predicted energy values with 45-degree reference line."""
        save_path = save_path or (self.figures_dir / "actual_vs_predicted.png")
        
        # Subsample if dataset is large to maintain crisp rendering
        if len(y_true) > sample_size:
            idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
            yt = y_true[idx]
            yp = y_pred[idx]
        else:
            yt, yp = y_true, y_pred

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(yt, yp, alpha=0.45, edgecolors="none", color="#1f77b4", label="Observations")
        
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Forecast (y = x)")

        ax.set_title(f"Actual vs Predicted Appliance Energy ({model_name})", fontsize=13, weight="bold")
        ax.set_xlabel("Actual Energy Consumption (Wh)", fontsize=11)
        ax.set_ylabel("Predicted Energy Consumption (Wh)", fontsize=11)
        ax.legend(frameon=True)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved actual vs predicted plot to {save_path}")
        return save_path

    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None
    ) -> Path:
        """Plots distribution of residuals (errors = y_true - y_pred)."""
        save_path = save_path or (self.figures_dir / "residual_distribution.png")
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax, bins=50)
        ax.axvline(0, color="red", linestyle="--", lw=1.8, label="Zero Error Line")
        ax.axvline(np.mean(residuals), color="black", linestyle=":", lw=1.8,
                   label=f"Mean Error: {np.mean(residuals):.2f} Wh")

        ax.set_title(f"Residual Distribution ({model_name})", fontsize=13, weight="bold")
        ax.set_xlabel("Residual (Actual - Predicted) [Wh]", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(frameon=True)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved residual distribution plot to {save_path}")
        return save_path

    def plot_prediction_error_over_time(
        self,
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
        window: int = 200
    ) -> Path:
        """Plots time-series timeline of actual vs predicted and absolute error."""
        save_path = save_path or (self.figures_dir / "error_over_time.png")
        
        # Display subset window for high temporal resolution
        dates_sub = pd.to_datetime(dates.iloc[:window])
        yt_sub = y_true[:window]
        yp_sub = y_pred[:window]
        err_sub = np.abs(yt_sub - yp_sub)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(dates_sub, yt_sub, label="Actual Consumption", color="#1f77b4", lw=1.8)
        ax1.plot(dates_sub, yp_sub, label=f"Predicted ({model_name})", color="#ff7f0e", linestyle="--", lw=1.8)
        ax1.set_ylabel("Appliances (Wh)", fontsize=11)
        ax1.set_title(f"Forecast Horizon Comparison over Time ({model_name})", fontsize=13, weight="bold")
        ax1.legend(loc="upper right", frameon=True)

        ax2.bar(dates_sub, err_sub, color="#d62728", alpha=0.6, width=0.005, label="Absolute Error (|y - y_hat|)")
        ax2.set_ylabel("Abs Error (Wh)", fontsize=11)
        ax2.set_xlabel("Timestamp", fontsize=11)
        ax2.legend(loc="upper right", frameon=True)

        plt.xticks(rotation=20)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved error over time plot to {save_path}")
        return save_path

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
        top_n: int = 15
    ) -> Path:
        """Plots top N feature importances."""
        save_path = save_path or (self.figures_dir / "feature_importance.png")
        
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        df_imp = df_imp.sort_values("Importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=df_imp, x="Importance", y="Feature", palette="Blues_r", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances ({model_name})", fontsize=13, weight="bold")
        ax.set_xlabel("Relative Importance", fontsize=11)
        ax.set_ylabel("Feature Name", fontsize=11)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved feature importance plot to {save_path}")
        return save_path

    def save_comparison_table(
        self,
        comparison_records: List[Dict[str, Any]],
        csv_path: Path = COMPARISON_CSV_PATH
    ) -> pd.DataFrame:
        """Saves tabular comparison of baseline and candidate ML models."""
        df_comp = pd.DataFrame(comparison_records)
        df_comp.to_csv(csv_path, index=False)
        logger.info(f"Saved model comparison table to {csv_path}")
        return df_comp

    def save_metrics_json(
        self,
        metrics_dict: Dict[str, Any],
        json_path: Path = METRICS_PATH
    ) -> None:
        """Saves evaluation results in a structured JSON file."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved metrics to {json_path}")
