from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import mlflow
import numpy as np

from data import load_data
from features import scale_features

# Load data
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = load_data()
X_train, X_test = scale_features(X_train, X_test)

# Classification baselines
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42)
}

# Regression baselines
regressors = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42)
}

mlflow.set_experiment("diabetes_project")

with mlflow.start_run(run_name="baselines"):
    # Classification
    for name, model in classifiers.items():
        model.fit(X_train, y_class_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_class_test, preds)
        f1 = f1_score(y_class_test, preds)
        mlflow.log_metric(f"{name}_Accuracy", acc)
        mlflow.log_metric(f"{name}_F1", f1)
        print(f"{name} -> Accuracy: {acc:.3f}, F1: {f1:.3f}")
    
    # Regression
    for name, model in regressors.items():
        model.fit(X_train, y_reg_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_reg_test, preds)
        rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
        mlflow.log_metric(f"{name}_MAE", mae)
        mlflow.log_metric(f"{name}_RMSE", rmse)
        print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
