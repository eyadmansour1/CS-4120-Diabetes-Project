from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(test_size=0.2, random_state=42):
    diabetes = datasets.load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    
    # Classification target: binary split on median progression
    median = y.median()
    y_class = (y > median).astype(int)
    
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=test_size, random_state=random_state
    )
    
    return (X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test)
