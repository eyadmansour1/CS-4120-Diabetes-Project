# run.py
# -*- coding: utf-8 -*-
"""
Week 8 midpoint pipeline (NO LEAKAGE) using TRUE Y.
Adds:
- Interactive plot display: each of the 4 figures pops up one-by-one (close to continue).
- Rich terminal logging: dataset summary, dtypes, split sizes, and the two metrics tables.
- MLflow autologging DISABLED to remove spurious warnings; single parent run handles manual logging.

Outputs exactly 4 plots + 2 tables and logs artifacts with MLflow.
"""

import os, re, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# ============= USER OPTIONS =============
SHOW_PLOTS = True   # Pop up each plot window sequentially
# ========================================

# Silence optional Git warning and reduce MLflow verbosity
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ARTIFACTS_DIR = Path("artifacts"); PLOTS_DIR = ARTIFACTS_DIR/"plots"; TABLES_DIR = ARTIFACTS_DIR/"tables"
for d in (ARTIFACTS_DIR, PLOTS_DIR, TABLES_DIR): d.mkdir(parents=True, exist_ok=True)

# -------------------- Loaders --------------------
def _try_read_csv(p: Path, **kw): return pd.read_csv(p, engine="python", **kw)

def _read_with_tab_variants(p: Path) -> pd.DataFrame:
    # Real tab
    try:
        df = _try_read_csv(p, sep="\t")
        if df.shape[1] > 1: return df
        if len(df.columns)==1 and "\\t" in str(df.columns[0]): raise ValueError
    except Exception: pass
    # Literal '\t'
    try:
        df = _try_read_csv(p, sep=r"\\t")
        if df.shape[1] > 1: return df
    except Exception: pass
    # Whitespace
    try:
        df = _try_read_csv(p, delim_whitespace=True)
        if df.shape[1] > 1: return df
    except Exception: pass
    # Replace literal '\t' with actual
    try:
        text = p.read_text(encoding="utf-8", errors="replace").replace("\\t", "\t")
        tmp = Path("_tmp_retabbed.csv"); tmp.write_text(text, encoding="utf-8")
        df = _try_read_csv(tmp, sep="\t"); tmp.unlink(missing_ok=True)
        if df.shape[1] > 1: return df
    except Exception: pass
    return _try_read_csv(p)

def load_dataset() -> pd.DataFrame:
    for fname, kind in [
        ("diabetes.xlsx", "excel"),
        ("diabetes.tab.csv", "tab"),
        ("diabetes.tsv", "tab"),
        ("diabetes.csv", "auto"),
    ]:
        p = Path(fname)
        if p.exists():
            if kind=="excel": df = pd.read_excel(p)
            elif kind=="tab": df = _read_with_tab_variants(p)
            else: df = _try_read_csv(p)
            print(f"[loader] Loaded {fname} shape={df.shape}")
            df.columns = [str(c).strip().strip('\'"') for c in df.columns]
            return df

    # Fallback (only if no file found)
    n = 768; rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "AGE": rng.integers(21, 81, n), "SEX": rng.integers(0,2,n),
        "BMI": rng.normal(32,7,n).clip(10), "BP": rng.normal(70,12,n).clip(0),
        "S1": rng.normal(size=n), "S2": rng.normal(size=n), "S3": rng.normal(size=n),
        "S4": rng.normal(size=n), "S5": rng.normal(size=n), "S6": rng.normal(size=n),
    })
    df["Y"] = (0.8*df["BMI"] + 0.5*df["AGE"] + 2.0*df["S5"] + rng.normal(0,10,n) + 200).astype(float)
    print(f"[loader] Using synthetic fallback shape={df.shape}")
    return df

# -------------------- Cleaning & coercion --------------------
def _normalize_colname(c: str) -> str:
    c = str(c).replace("\xa0"," ").strip().strip('\'"')
    c = re.sub(r"\s+"," ",c)
    return c.replace(" ","_")

_NUM_KEEP_RE = re.compile(r"[^0-9eE\+\-\.]")  # drop anything NOT numeric-related

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    st = s.astype(str).str.strip().str.strip('\'"')
    st = st.str.replace(",","",regex=False).str.replace("%","",regex=False)
    st = st.apply(lambda x: _NUM_KEEP_RE.sub("", x))
    return pd.to_numeric(st, errors="coerce")

def clean_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [_normalize_colname(c) for c in df.columns]
    empty = [c for c in df.columns if df[c].isna().all() or (df[c].astype(str).str.strip()=="").all()]
    if empty: df = df.drop(columns=empty)
    for c in df.columns:
        if df[c].dtype == "O":
            coerced = _coerce_numeric_series(df[c])
            if coerced.notna().mean() >= 0.7 or c.upper()=="Y":
                df[c] = coerced
    return df

def force_y_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "Y" not in df.columns: raise ValueError("Column 'Y' not found.")
    if not pd.api.types.is_numeric_dtype(df["Y"]): df["Y"] = _coerce_numeric_series(df["Y"])
    valid_ratio = df["Y"].notna().mean()
    if valid_ratio < 0.98:
        raise ValueError(f"'Y' could not be coerced to numeric (valid={valid_ratio:.2%}). "
                         f"Please inspect your source file for non-numeric Y values.")
    df = df.dropna(subset=["Y"]).reset_index(drop=True)
    return df

# -------------------- Split & Preprocess --------------------
def split_train_val_test_for_Y(df: pd.DataFrame):
    df = df.copy()
    global_thr = df["Y"].median()
    df["Outcome"] = (df["Y"] >= global_thr).astype(int)

    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=RANDOM_STATE,
                                           stratify=df["Outcome"])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_STATE,
                                         stratify=df.loc[temp_idx,"Outcome"])

    train_df, val_df, test_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    thr = train_df["Y"].median()
    for part in (train_df, val_df, test_df):
        part["Outcome"] = (part["Y"] >= thr).astype(int)
    return train_df, val_df, test_df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if X[c].dtype=="O"]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
    trs = []
    if num: trs.append(("num", num_pipe, num))
    if cat: trs.append(("cat", cat_pipe, cat))
    if not trs: raise ValueError("No usable features after cleaning.")
    return ColumnTransformer(trs, remainder="drop")

# -------------------- Plot helpers (save + optional pop-up) --------------------
def _finalize_plot(path: Path, title: str):
    plt.tight_layout(); plt.savefig(path, dpi=200)
    if SHOW_PLOTS:
        try: plt.gcf().canvas.manager.set_window_title(title)
        except Exception: pass
        plt.show(block=True)
    plt.close()

def plot_target_distribution(df_all, path: Path):
    plt.figure(figsize=(5,4))
    sns.countplot(x="Outcome", data=df_all)
    plt.title("Target distribution (classification)")
    plt.xlabel("Class"); plt.ylabel("Count")
    _finalize_plot(path, "Target distribution")

def plot_correlation_heatmap(df_all, path: Path):
    feats = df_all.drop(columns=["Outcome"], errors="ignore").select_dtypes(include=[np.number])
    plt.figure(figsize=(7,6))
    if feats.shape[1]==0:
        plt.text(0.5,0.5,"No numeric features.",ha="center"); plt.axis("off")
    else:
        corr = feats.corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric features)")
    _finalize_plot(path, "Correlation heatmap")

def confusion_matrix_plot(y_true, y_pred, path: Path, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(title)
    _finalize_plot(path, "Confusion matrix")

def residuals_plot(y_true, y_pred, path: Path, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6, s=16)
    plt.axhline(0, ls="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(title)
    _finalize_plot(path, "Residuals vs Predicted")

# -------------------- Metrics --------------------
def evaluate_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred, zero_division=0); roc = np.nan
    if y_proba is not None:
        try: roc = roc_auc_score(y_true, y_proba)
        except Exception: roc = np.nan
    return {"accuracy": acc, "f1": f1, "roc_auc": roc}

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred); rmse = float(np.sqrt(mse))
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": rmse, "R2": r2_score(y_true, y_pred)}

# -------------------- Terminal helpers --------------------
def print_dataset_summary(df, train_df, val_df, test_df):
    print("\n===== DATASET SUMMARY =====")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Dtypes:")
    dts = df.dtypes.reset_index()
    dts.columns = ["column", "dtype"]
    print(dts.to_string(index=False))
    print("\nHead (first 5 rows):")
    with pd.option_context('display.max_columns', None, 'display.width', 140):
        print(df.head(5).to_string(index=False))
    print("\nSplit sizes:")
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

def print_table(title, df):
    print(f"\n===== {title} =====")
    with pd.option_context('display.float_format', lambda x: f"{x:0.4f}"):
        print(df.to_string(index=False))

# -------------------- Main --------------------
def main():
    df_raw = load_dataset()
    print("[loader] Raw columns:", df_raw.columns.tolist())

    df = clean_and_coerce(df_raw)
    df = force_y_numeric(df)  # ensures 'Y' is numeric

    # Leakage-free split using Y
    train_df, val_df, test_df = split_train_val_test_for_Y(df)

    # Terminal: dataset info
    print_dataset_summary(df, train_df, val_df, test_df)

    # Features = all except Y and Outcome
    df_all = pd.concat([train_df, val_df, test_df], axis=0)
    feature_cols = [c for c in df_all.columns if c not in ["Y", "Outcome"]]
    if not feature_cols: raise ValueError("No features left after target construction.")

    X_train, X_val, X_test = train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]
    y_train_clf, y_val_clf, y_test_clf = train_df["Outcome"].values, val_df["Outcome"].values, test_df["Outcome"].values
    y_train_reg, y_val_reg, y_test_reg = train_df["Y"].values,       val_df["Y"].values,       test_df["Y"].values

    # EDA plots (1,2)
    plot_target_distribution(df_all, PLOTS_DIR/"plot1_target_distribution.png")
    plot_correlation_heatmap(df_all, PLOTS_DIR/"plot2_corr.png")

    preprocessor = build_preprocessor(X_train)

    # --- MLflow setup: disable autologging and use a clean local tracking dir ---
    mlflow.autolog(disable=True)  # remove autolog side-effects/warnings
    mlflow.set_tracking_uri("file:./mlruns_clean")
    mlflow.set_experiment("midpoint_diabetes")

    # Single parent run to contain everything (metrics + artifacts)
    with mlflow.start_run(run_name="midpoint_full_run", nested=False):

        # ---- Classification baselines ----
        classifiers = {
            "logreg":   LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
            "dtree_clf": DecisionTreeClassifier(random_state=RANDOM_STATE),
        }
        cls_rows, best_name, best_val_score, best_test_preds = [], None, -np.inf, None

        for name, model in classifiers.items():
            with mlflow.start_run(run_name=f"classification__{name}", nested=True):
                pipe = Pipeline([("prep", preprocessor), ("model", model)]); pipe.fit(X_train, y_train_clf)

                y_val_pred = pipe.predict(X_val)
                y_val_proba = None
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    y_val_proba = pipe.predict_proba(X_val)[:,1]
                elif hasattr(pipe.named_steps["model"], "decision_function"):
                    s = pipe.decision_function(X_val); m,M = s.min(), s.max(); y_val_proba = (s-m)/(M-m+1e-9)
                val = evaluate_classification(y_val_clf, y_val_pred, y_val_proba)

                y_test_pred = pipe.predict(X_test)
                y_test_proba = None
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    y_test_proba = pipe.predict_proba(X_test)[:,1]
                elif hasattr(pipe.named_steps["model"], "decision_function"):
                    s = pipe.decision_function(X_test); m,M = s.min(), s.max(); y_test_proba = (s-m)/(M-m+1e-9)
                test = evaluate_classification(y_test_clf, y_test_pred, y_test_proba)

                row = {
                    "model": name,
                    "val_accuracy": val["accuracy"], "val_f1": val["f1"], "val_roc_auc": val["roc_auc"],
                    "test_accuracy": test["accuracy"], "test_f1": test["f1"], "test_roc_auc": test["roc_auc"],
                }
                cls_rows.append(row)

                # manual logging
                for k,v in row.items():
                    if k!="model" and v is not None and not (isinstance(v,float) and np.isnan(v)):
                        mlflow.log_metric(k, float(v))
                mlflow.log_params({"task":"classification","model_name":name})

                score = (val["roc_auc"] if not np.isnan(val["roc_auc"]) else 0.0) + 1e-3*val["f1"]
                if score > best_val_score: best_val_score, best_name, best_test_preds = score, name, y_test_pred

        if best_name is not None:
            confusion_matrix_plot(y_test_clf, best_test_preds, PLOTS_DIR/"plot3_confusion_best_clf.png",
                                  f"Confusion Matrix — Test ({best_name})")

        cls_df = pd.DataFrame(cls_rows)[[
            "model","val_accuracy","val_f1","val_roc_auc","test_accuracy","test_f1","test_roc_auc"
        ]]
        cls_df.to_csv(TABLES_DIR/"classification_metrics.csv", index=False)
        print_table("CLASSIFICATION METRICS (val/test)", cls_df)

        # ---- Regression baselines ----
        regressors = {
            "linreg":   LinearRegression(),
            "dtree_reg": DecisionTreeRegressor(random_state=RANDOM_STATE),
        }
        reg_rows, best_reg_name, best_val_rmse, best_reg_test_pred = [], None, np.inf, None

        for name, model in regressors.items():
            with mlflow.start_run(run_name=f"regression__{name}", nested=True):
                pipe = Pipeline([("prep", preprocessor), ("model", model)]); pipe.fit(X_train, y_train_reg)

                y_val_pred = pipe.predict(X_val); val = evaluate_regression(y_val_reg, y_val_pred)
                y_test_pred = pipe.predict(X_test); test = evaluate_regression(y_test_reg, y_test_pred)

                row = {"model":name, "val_MAE":val["MAE"], "val_RMSE":val["RMSE"], "val_R2":val["R2"],
                       "test_MAE":test["MAE"], "test_RMSE":test["RMSE"], "test_R2":test["R2"]}
                reg_rows.append(row)

                for k,v in row.items():
                    if k!="model": mlflow.log_metric(k, float(v))
                mlflow.log_params({"task":"regression","model_name":name})

                if val["RMSE"] < best_val_rmse:
                    best_val_rmse, best_reg_name, best_reg_test_pred = val["RMSE"], name, y_test_pred

        if best_reg_name is not None:
            residuals_plot(y_test_reg, best_reg_test_pred, PLOTS_DIR/"plot4_residuals_best_reg.png",
                           f"Residuals vs Predicted — Test ({best_reg_name})")

        reg_df = pd.DataFrame(reg_rows)[[
            "model","val_MAE","val_RMSE","val_R2","test_MAE","test_RMSE","test_R2"
        ]]
        reg_df.to_csv(TABLES_DIR/"regression_metrics.csv", index=False)
        print_table("REGRESSION METRICS (val/test)", reg_df)

        # Notes
        with open(ARTIFACTS_DIR/"notes.txt","w",encoding="utf-8") as f:
            f.write("Targets: Regression uses TRUE Y; Outcome = (Y >= train median) computed after split (no leakage).\n"
                    "Split: 60/20/20 stratified on provisional Outcome; seed=42.\n"
                    "Preprocessing: numeric median-impute + StandardScaler; categorical most_frequent + OneHot.\n"
                    "Baselines: LogReg/DT for classification; LinearReg/DTReg for regression.\n")

        # Log artifacts once under the parent run
        for p in [TABLES_DIR/"classification_metrics.csv", TABLES_DIR/"regression_metrics.csv",
                  PLOTS_DIR/"plot1_target_distribution.png", PLOTS_DIR/"plot2_corr.png",
                  PLOTS_DIR/"plot3_confusion_best_clf.png", PLOTS_DIR/"plot4_residuals_best_reg.png",
                  ARTIFACTS_DIR/"notes.txt"]:
            if p.exists(): mlflow.log_artifact(str(p))

    print("\n[OK] Artifacts written to ./artifacts and logged to MLflow (autologging disabled).")

if __name__ == "__main__":
    main()
