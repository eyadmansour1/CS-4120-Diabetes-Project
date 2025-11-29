# src/features.py
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_val, X_test):
    """
    Fit scaler on TRAIN only, then transform val and test.
    Returns scaled arrays and the fitted scaler (for reuse / logging).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
