import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_CSV_PATH = os.path.join("dataset", "telco_customer_churn.csv")

def acquire_dataset(target_path: str = LOCAL_CSV_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        print(f"[Stage 1: Acquire] Downloading dataset from {DATASET_URL}...")
        resp = requests.get(DATASET_URL, timeout=30)
        with open(target_path, "wb") as f:
            f.write(resp.content)
        print(f"[Stage 1: Acquire] Saved dataset to {target_path}")
    else:
        print(f"[Stage 1: Acquire] Local dataset found at {target_path}")
    
    df = pd.read_csv(target_path)
    print(f"[Stage 1: Acquire] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def detect_outliers_iqr(df: pd.DataFrame, continuous_cols: list) -> dict:
    outlier_stats = {}
    for col in continuous_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_stats[col] = {
            "Q1": float(q1),
            "Q3": float(q3),
            "IQR": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outliers)),
            "outlier_pct": float(len(outliers) / len(df) * 100),
        }
    return outlier_stats

def clean_and_preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Step 1: Whitespace & Missing Values Handling
    blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
    blank_count = int(blank_mask.sum())
    print(f"[Stage 2: Parse] Found {blank_count} blank entries in TotalCharges.")
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Step 2: Duplicates Check
    dup_count = int(df.duplicated().sum())
    print(f"[Stage 3: Filter] Duplicate rows detected: {dup_count}")

    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    outlier_stats = detect_outliers_iqr(df, continuous_cols)

    df_clean = df.copy()

    # Step 3: Feature Encoding
    df["Churn_Numeric"] = (df["Churn"] == "Yes").astype(int)

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col == "gender":
            df[col] = (df[col] == "Male").astype(int)
        else:
            df[col] = (df[col] == "Yes").astype(int)

    multiclass_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    df_encoded = pd.get_dummies(df, columns=multiclass_cols, drop_first=True)
    
    feature_cols = [c for c in df_encoded.columns if c not in ["customerID", "Churn", "Churn_Numeric"]]
    X = df_encoded[feature_cols].astype(np.float32)
    y = df_encoded["Churn_Numeric"].values.astype(np.float32)

    # Step 4: Stratified Splitting (Train: 70%, Validation: 15%, Test: 15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    # Step 5: Feature Normalization / Standardization
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_val_scaled[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    metadata = {
        "raw_shape": df_raw.shape,
        "clean_shape": df_clean.shape,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "num_features": X.shape[1],
        "feature_names": feature_cols,
        "continuous_cols": continuous_cols,
        "outlier_stats": outlier_stats,
        "blank_imputed": blank_count,
        "duplicates": dup_count,
        "target_balance": {
            "No": int((y == 0).sum()),
            "Yes": int((y == 1).sum()),
            "Churn_Rate": float(y.mean() * 100),
        },
    }

    print(f"[Split & Scale] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | Features: {X.shape[1]}")

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "metadata": metadata,
    }

if __name__ == "__main__":
    df = acquire_dataset()
    data = clean_and_preprocess(df)
    print("Preprocessed successfully. Features count:", data["metadata"]["num_features"])
