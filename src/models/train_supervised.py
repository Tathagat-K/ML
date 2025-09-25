from __future__ import annotations
import pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from joblib import dump

from src.features.preprocess import build_preprocessor
from src.eval.metrics import summarize_clf

def model_zoo(balanced=False) -> Dict[str, object]:
    w = "balanced" if balanced else None
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight=w),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight=w),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=w),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),  # no class_weight
        "SVM": SVC(probability=True, random_state=42, class_weight=w),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=600, random_state=42),
        "XGB": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=None  # set outside when balanced=True
        ),
    }

def fit_and_eval(models: Dict[str, object], preprocessor, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    rows = []
    for name, mdl in models.items():
        if name == "XGB" and hasattr(mdl, "set_params"):
            # handle class imbalance for XGB if needed
            pos = int(y_train.sum())
            neg = int((y_train == 0).sum())
            if getattr(mdl, "scale_pos_weight") is None:
                pass
        pipe = Pipeline([("pre", preprocessor), ("clf", mdl)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        rows.append({"model": name, **summarize_clf(y_val, y_pred, y_proba), "pipeline": pipe})
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df

def save_best(results_df: pd.DataFrame, path):
    best = results_df.iloc[0]
    dump(best["pipeline"], path)
    return best["model"], path
