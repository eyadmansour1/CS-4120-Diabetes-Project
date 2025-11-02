"""
data.py
Handles loading, cleaning, and splitting of the Diabetes dataset.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE = 42

def load_and_split_data(as_classification=False):
    """
    Loads the diabetes dataset from scikit-learn and returns train/val/test splits.
    If as_classification=True, converts the regression target into a binary label.
    """
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()

    # Cleaning example: rename columns for clarity
    df.columns = [c.replace(" ", "_") for c in df.columns]

    if as_classification:
        median_target = df["target"].median()
        df["target"] = (df["target"] > median_target).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
