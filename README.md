
# Diabetes Progression Prediction — CS-4120 Final Project
### Group Diabetes — Fall 2025  
**Members:** Eyad Mansour, Shehab Abugharsa  
**Course:** CS-4120 Machine Learning  
**Instructor:** Dr. Dania Tamayo-Vera  

---

## 📌 Project Overview

This project explores machine learning methods for predicting diabetes disease progression using the **scikit-learn Diabetes dataset**.  
We approach the problem through two supervised learning tasks:

1. **Binary Classification** — Determine whether a patient's disease progression is above or below the median.  
2. **Regression** — Predict the continuous disease progression score.

We implement a complete machine learning pipeline including data preprocessing, classical machine learning models, neural networks, model evaluation, feature importance, visualizations, and reproducibility measures.

---

## 📂 Repository Structure

```
.
├── src/
│   ├── data.py                   # Dataset loading & unified stratified train/val/test splitting
│   ├── features.py               # Feature scaling using StandardScaler (fit on train only)
│   ├── evaluate.py               # Metric functions & plot utilities
│   ├── train_baselines.py        # Classical ML models (classification & regression)
│   ├── train_nn.py               # Neural network models (MLP classifiers & regressors)
│   ├── feature_importance.py     # Permutation-based feature importance plot
│   ├── make_tables.py            # Metric summary tables for classical models
│   ├── utils.py                  # Random seed utility for reproducibility
│
├── plots/                        # All generated figures & visualizations
├── tables/                       # Classical model performance tables (CSV)
├── mlruns/                       # MLflow experiment logs
│
├── requirements.txt              # Dependency versions for reproducibility
└── Final_Report_GroupDiabetes_FULL.docx   # Full final project report
```

---

## 🚀 How to Run the Project

Install dependencies first:

```bash
pip install -r requirements.txt
```

### 1. Train Classical Models
```bash
python src/train_baselines.py
```

### 2. Train Neural Networks
```bash
python src/train_nn.py
```

### 3. Generate Feature Importance Plot
```bash
python src/feature_importance.py
```

### 4. Generate Tables for Classical Models
```bash
python src/make_tables.py
```

---

## 📊 Final Results Summary

### Classification Performance
| Model | Test Accuracy | Test F1 |
|-------|---------------|---------|
| Logistic Regression | **0.775** | **0.775** |
| Neural Network (MLP) | 0.753 | 0.753 |
| Decision Tree Classifier | 0.640 | 0.640 |

### Regression Performance
| Model | Test MAE | Test RMSE |
|--------|-----------|------------|
| Linear Regression | **43.27** | **53.22** |
| Neural Network (MLP) | 46.40 | 59.83 |
| Decision Tree Regressor | 55.94 | 74.81 |

---

## 🔧 Reproducibility

- Unified train/validation/test splitting  
- Scaling fit only on training data  
- Random seed initialization  
- MLflow experiment tracking  
- Pinned dependencies in `requirements.txt`

---

## 📑 Final Report

All analysis, plots, interpretations, and tables are included in:

```
Final_Report_GroupDiabetes_FULL.docx
```

---

## 📬 Contact

For questions or reproduction issues, please contact:  
**Eyad Mansour & Shehab Abugharsa**


