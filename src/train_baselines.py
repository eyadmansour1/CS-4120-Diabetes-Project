"""
train_baselines.py
Trains classical ML baselines for both regression and classification.
Logs metrics with MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from src.data import load_and_split_data
from src.features import scale_features

def train_regression_models():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(as_classification=False)
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42)
    }

    mlflow.set_experiment("diabetes_midpoint_regression")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)

            mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
            mlflow.sklearn.log_model(model, name)

def train_classification_models():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(as_classification=True)
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42)
    }

    mlflow.set_experiment("diabetes_midpoint_classification")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            y_prob = model.predict_proba(X_val_s)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            roc = roc_auc_score(y_val, y_prob)

            mlflow.log_metrics({"Accuracy": acc, "F1": f1, "ROC_AUC": roc})
            mlflow.sklearn.log_model(model, name)

if __name__ == "__main__":
    train_regression_models()
    train_classification_models()
