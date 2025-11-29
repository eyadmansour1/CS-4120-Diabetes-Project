
"""
Train classical models and export metrics tables as CSV.

This script does NOT use MLflow. It simply re-trains the classical
models using the same split and writes:

  tables/classical_classification_results.csv
  tables/classical_regression_results.csv

You can then add your NN metrics as extra rows in those CSVs.
"""

import sys
from pathlib import Path

# Allow "from src.xxx import ..." when running as a script
sys.path.append(str(Path(__file__).parent.parent))

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.data import split_for_both_tasks
from src.features import scale_features
from src.evaluate import classification_metrics, regression_metrics

TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(exist_ok=True)


def build_classification_table():
    (
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        _, _, _
    ) = split_for_both_tasks()

    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    ]

    rows = []

    for name, model in models:
        model.fit(X_train_s, y_train_clf)

        for split_name, X_split, y_split in [
            ("val", X_val_s, y_val_clf),
            ("test", X_test_s, y_test_clf),
        ]:
            y_pred = model.predict(X_split)
            metrics = classification_metrics(y_split, y_pred)
            rows.append(
                {
                    "Model": name,
                    "Split": split_name,
                    "Accuracy": metrics["accuracy"],
                    "F1_macro": metrics["f1_macro"],
                }
            )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "classical_classification_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved classification table to: {out_path}")


def build_regression_table():
    (
        X_train, X_val, X_test,
        _, _, _,
        y_train_reg, y_val_reg, y_test_reg
    ) = split_for_both_tasks()

    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    models = [
        ("LinearRegression", LinearRegression()),
        ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
    ]

    rows = []

    for name, model in models:
        model.fit(X_train_s, y_train_reg)

        for split_name, X_split, y_split in [
            ("val", X_val_s, y_val_reg),
            ("test", X_test_s, y_test_reg),
        ]:
            y_pred = model.predict(X_split)
            metrics = regression_metrics(y_split, y_pred)
            rows.append(
                {
                    "Model": name,
                    "Split": split_name,
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                }
            )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "classical_regression_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved regression table to: {out_path}")


def main():
    build_classification_table()
    build_regression_table()
    print("Done building tables.")


if __name__ == "__main__":
    main()
