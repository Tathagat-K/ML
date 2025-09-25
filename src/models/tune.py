from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

def tune_random_forest(preprocessor, X_train, y_train, n_iter=20, random_state=42):
    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    param = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4]
    }
    pipe = Pipeline([("pre", preprocessor), ("clf", rf)])
    search = RandomizedSearchCV(
        pipe, param, n_iter=n_iter, cv=3, scoring="roc_auc", n_jobs=-1, random_state=random_state, refit=True
    )
    search.fit(X_train, y_train)
    return search  # .best_estimator_ is a full pipeline
