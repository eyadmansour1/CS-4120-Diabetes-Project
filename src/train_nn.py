# src/train_nn.py
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow running script directly
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.data import split_for_both_tasks
from src.features import scale_features
from src.evaluate import (
    classification_metrics,
    regression_metrics,
    plot_learning_curve,
)
from src.utils import set_seed

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, task_type="clf"):
        super().__init__()
        hidden = 64
        self.task_type = task_type
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.out(x)
        if self.task_type == "clf":
            return torch.sigmoid(x)
        return x  # regression

def to_loader(X, y, batch_size=32, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_nn_classification(epochs=50, lr=1e-3, batch_size=32):
    set_seed(42)
    (
        X_train, X_val, X_test,
        y_train_clf, y_val_clf, y_test_clf,
        _, _, _
    ) = split_for_both_tasks()

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    input_dim = X_train_s.shape[1]
    model = MLP(input_dim, output_dim=1, task_type="clf").to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = to_loader(X_train_s, y_train_clf, batch_size=batch_size, shuffle=True)
    val_loader   = to_loader(X_val_s, y_val_clf, batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}

    with mlflow.start_run(run_name="nn_classification"):
        mlflow.log_param("model", "MLP")
        mlflow.log_param("task", "classification")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.mean(train_losses))

            # validation
            model.eval()
            val_losses = []
            y_val_all, y_val_pred_all = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_losses.append(loss.item())
                    y_val_all.append(yb.cpu().numpy())
                    y_val_pred_all.append(preds.cpu().numpy())
            val_loss = float(np.mean(val_losses))
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            y_val_all = np.vstack(y_val_all).ravel()
            y_val_pred_all = np.vstack(y_val_pred_all).ravel()
            y_val_labels = (y_val_pred_all >= 0.5).astype(int)
            metrics = classification_metrics(y_val_all, y_val_labels)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_f1_macro", metrics["f1_macro"], step=epoch)

        # final test evaluation
        model.eval()
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds_test = model(X_test_t).cpu().numpy().ravel()
        y_test_labels = (preds_test >= 0.5).astype(int)
        test_metrics = classification_metrics(y_test_clf, y_test_labels)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # learning curve plot
        lc_path = PLOTS_DIR / "nn_classification_learning_curve.png"
        plot_learning_curve(
            history,
            title="NN Classification Learning Curve",
            metric_name="Loss",
            path=lc_path,
        )
        mlflow.log_artifact(str(lc_path))

        mlflow.pytorch.log_model(model, artifact_path="model")

    return history, test_metrics

def train_nn_regression(epochs=50, lr=1e-3, batch_size=32):
    set_seed(42)
    (
        X_train, X_val, X_test,
        _, _, _,
        y_train_reg, y_val_reg, y_test_reg
    ) = split_for_both_tasks()

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    input_dim = X_train_s.shape[1]
    model = MLP(input_dim, output_dim=1, task_type="reg").to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = to_loader(X_train_s, y_train_reg, batch_size=batch_size, shuffle=True)
    val_loader   = to_loader(X_val_s, y_val_reg, batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}

    with mlflow.start_run(run_name="nn_regression"):
        mlflow.log_param("model", "MLP")
        mlflow.log_param("task", "regression")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.mean(train_losses))

            model.eval()
            val_losses = []
            y_val_all, y_val_pred_all = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_losses.append(loss.item())
                    y_val_all.append(yb.cpu().numpy())
                    y_val_pred_all.append(preds.cpu().numpy())
            val_loss = float(np.mean(val_losses))
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            y_val_all = np.vstack(y_val_all).ravel()
            y_val_pred_all = np.vstack(y_val_pred_all).ravel()
            metrics = regression_metrics(y_val_all, y_val_pred_all)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_mae", metrics["mae"], step=epoch)
            mlflow.log_metric("val_rmse", metrics["rmse"], step=epoch)

        # test evaluation
        model.eval()
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds_test = model(X_test_t).cpu().numpy().ravel()
        test_metrics = regression_metrics(y_test_reg, preds_test)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        lc_path = PLOTS_DIR / "nn_regression_learning_curve.png"
        plot_learning_curve(
            history,
            title="NN Regression Learning Curve",
            metric_name="Loss",
            path=lc_path,
        )
        mlflow.log_artifact(str(lc_path))

        mlflow.pytorch.log_model(model, artifact_path="model")

    return history, test_metrics

if __name__ == "__main__":
    mlflow.set_experiment("diabetes_nn")
    print("Training NN classification...")
    hist_clf, test_clf = train_nn_classification()
    print("Training NN regression...")
    hist_reg, test_reg = train_nn_regression()
    print("Done.")
