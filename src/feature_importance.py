# src/feature_importance.py
"""
Compute and plot feature importance for a classical model
(using permutation importance) to satisfy the Final report
"Feature importance or ablation" plot requirement.

We use a Logistic Regression classifier on the diabetes dataset.
"""

import sys
from pathlib import Path

# Allow "from src.xxx import ..." when running as a script
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_diabetes

from src.data import split_for_both_tasks
from src.features import scale_features

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def main():
    # --- 1. Get the SAME split used everywhere else ---
    (
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        _, _, _
    ) = split_for_both_tasks()

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # --- 2. Train a classical classifier ---
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train_clf)

    # --- 3. Permutation importance on validation set ---
    print("Computing permutation importance (this may take a moment)...")
    result = permutation_importance(
        clf,
        X_val_s,
        y_val_clf,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    # Get feature names from sklearn's diabetes dataset
    data = load_diabetes()
    feature_names = np.array(data.feature_names)

    # Sort by importance
    idx_sorted = np.argsort(importances_mean)[::-1]
    feature_names_sorted = feature_names[idx_sorted]
    importances_sorted = importances_mean[idx_sorted]
    std_sorted = importances_std[idx_sorted]

    # --- 4. Plot as horizontal bar chart ---
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(feature_names_sorted))
    plt.barh(y_pos, importances_sorted, xerr=std_sorted, align="center")
    plt.yticks(y_pos, feature_names_sorted)
    plt.gca().invert_yaxis()  # largest at top
    plt.xlabel("Mean Permutation Importance (decrease in score)")
    plt.title("Feature Importance (Logistic Regression, Classification)")
    plt.tight_layout()

    out_path = PLOTS_DIR / "feature_importance_classification.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance plot to: {out_path}")


if __name__ == "__main__":
    main()
