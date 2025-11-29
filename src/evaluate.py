"""
evaluate.py
Handles metrics and result visualization — confusion matrix, residuals, curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)


# ---------- METRIC HELPERS ----------

def classification_metrics(y_true, y_pred):
    """
    Return Accuracy and macro F1 as a dict.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def regression_metrics(y_true, y_pred):
    """
    Return MAE and RMSE as a dict.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # NOTE: squared=False is deprecated in scikit-learn 1.4+ and removed in 1.6+
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"mae": mae, "rmse": rmse}


# ---------- PLOTS ----------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals vs Predicted"):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(history, title, metric_name, path=None):
    """
    Plot train/val curves over epochs.

    history: dict with keys 'train' and 'val' (lists of scalar metrics).
    """
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"], label=f"Train {metric_name}")
    plt.plot(epochs, history["val"], label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
