"""
train_baselines.py
Trains classical ML baselines for both regression and classification.
Logs metrics with MLflow and produces plots.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow running script directly
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.data import load_and_split_data
from src.features import scale_features
from src.evaluate import (
    plot_confusion_matrix,
    plot_residuals,
    regression_metrics,
    classification_metrics,
)
from src.utils import set_seed

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def train_regression_models():
    set_seed(42)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        as_classification=False
    )
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    }

    mlflow.set_experiment("diabetes_regression_baselines")

    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            model.fit(X_train_s, y_train)

            # Validation and test predictions
            y_val_pred = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            val_metrics = regression_metrics(y_val, y_val_pred)
            test_metrics = regression_metrics(y_test, y_test_pred)

            # Log metrics with prefixes
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Residuals plot on test set
            res_path = PLOTS_DIR / f"residuals_{name}.png"
            plot_residuals(
                y_test,
                y_test_pred,
                title=f"Residuals vs Predicted - {name}",
            )
            # If you want to save instead of show, modify plot_residuals to accept path

            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append((name, val_metrics, test_metrics))

    return results


def train_classification_models():
    set_seed(42)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        as_classification=True
    )
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    }

    mlflow.set_experiment("diabetes_classification_baselines")

    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            model.fit(X_train_s, y_train)

            y_val_pred = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            val_metrics = classification_metrics(y_val, y_val_pred)
            test_metrics = classification_metrics(y_test, y_test_pred)

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            cm_path = PLOTS_DIR / f"confusion_{name}.png"
            plot_confusion_matrix(
                y_test,
                y_test_pred,
                title=f"Confusion Matrix - {name}",
            )
            # Same comment: you can modify plot_confusion_matrix to save instead of just show

            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append((name, val_metrics, test_metrics))

    return results


if __name__ == "__main__":
    print("Training regression baselines...")
    reg_results = train_regression_models()
    print("Training classification baselines...")
    clf_results = train_classification_models()
    print("Done.")
