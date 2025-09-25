# src/features/preprocess.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CHURN_LEAK_COLS = ["Churn Label", "Churn Value", "Churn Score", "Churn Reason"]
NOISE_COLS = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Phone Service"]

DEFAULT_NUMERIC = ["Monthly Charges", "Total Charges", "CLTV", "Tenure Months"]

@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

def _sanitize(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Strip whitespace, convert empty strings to NaN, and coerce numerics."""
    df = df.copy()

    # 1) strip whitespace on object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 2) empty strings -> NaN
    df[obj_cols] = df[obj_cols].replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "na": np.nan, "n/a": np.nan, "None": np.nan})

    # 3) coerce numeric columns to real numbers
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def make_Xy(df: pd.DataFrame,
            numeric_cols: List[str] = None,
            drop_noise: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Return X, y and the auto-detected numeric/categorical column lists."""
    if numeric_cols is None:
        numeric_cols = [c for c in DEFAULT_NUMERIC if c in df.columns]

    # sanitize BEFORE building X/y
    df = _sanitize(df, numeric_cols)

    y = df["Churn Value"].astype(int)
    drop_cols = [c for c in (CHURN_LEAK_COLS + (NOISE_COLS if drop_noise else [])) if c in df.columns]
    X = df.drop(columns=drop_cols)

    # keep only numeric_cols that are in X; all others considered categorical
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols

def split(X, y, test_size=0.3, val_size=0.5, random_state=42) -> Split:
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # NOTE: if your sklearn < 1.2, use OneHotEncoder(..., sparse=False)
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols)
        ]
    )
    return pre
