"""
data.py
Handles loading, cleaning, and splitting of the Diabetes dataset.
Uses ONE unified split for both classification and regression.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2  # => train = 0.6, val = 0.2, test = 0.2


def _load_diabetes_dataframe():
    """Load scikit-learn diabetes dataset as DataFrame with both targets."""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["y_reg"] = data.target  # continuous disease progression

    # Binary label for classification: 1 if >= median target, else 0
    median = df["y_reg"].median()
    df["y_clf"] = (df["y_reg"] >= median).astype(int)
    return df


def split_for_both_tasks(random_state: int = RANDOM_STATE):
    """
    Create a single train/val/test split and return features + both targets.

    Returns:
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        y_train_reg, y_val_reg, y_test_reg
    """
    df = _load_diabetes_dataframe()
    feature_cols = [c for c in df.columns if c not in ["y_reg", "y_clf"]]

    X = df[feature_cols].values
    y_reg = df["y_reg"].values
    y_clf = df["y_clf"].values

    idx = np.arange(len(df))

    # First: train vs temp (val+test), stratified on classification label
    test_size_total = VAL_RATIO + TEST_RATIO
    train_idx, temp_idx, y_train_clf, y_temp_clf = train_test_split(
        idx,
        y_clf,
        test_size=test_size_total,
        random_state=random_state,
        stratify=y_clf,
    )

    # Second: val vs test from temp, stratified on temp labels
    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx, y_val_clf, y_test_clf = train_test_split(
        temp_idx,
        y_temp_clf,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=y_temp_clf,
    )

    def take(indices):
        return (
            X[indices],
            y_clf[indices],
            y_reg[indices],
        )

    X_train, y_train_clf, y_train_reg = take(train_idx)
    X_val,   y_val_clf,   y_val_reg   = take(val_idx)
    X_test,  y_test_clf,  y_test_reg  = take(test_idx)

    return (
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        y_train_reg, y_val_reg, y_test_reg
    )


def load_and_split_data(as_classification=False, random_state: int = RANDOM_STATE):
    """
    Backwards-compatible wrapper used by train_baselines.py.
    Returns train/val/test for EITHER classification or regression.
    """
    (
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        y_train_reg, y_val_reg, y_test_reg
    ) = split_for_both_tasks(random_state=random_state)

    if as_classification:
        return X_train, X_val, X_test, y_train_clf, y_val_clf, y_test_clf
    else:
        return X_train, X_val, X_test, y_train_reg, y_val_reg, y_test_reg

