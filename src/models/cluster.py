from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

from src.features.preprocess import build_preprocessor, CHURN_LEAK_COLS

def prepare_unsupervised(df, preprocessor):
    X = df.drop(columns=[c for c in CHURN_LEAK_COLS if c in df.columns])
    Xt = preprocessor.fit_transform(X)
    return X, Xt

def choose_k_by_silhouette(Xt, k_min=2, k_max=9, random_state=42):
    scores = {}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(Xt)
        scores[k] = silhouette_score(Xt, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores

def kmeans_clusters(Xt, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(Xt)
    return labels, km

def evaluate_clustering(labels, y_true=None):
    out = {"n_clusters": int(len(np.unique(labels)))}
    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    return out

def pca_plot(Xt, labels=None, title="PCA"):
    p = PCA(n_components=2, random_state=42)
    Z = p.fit_transform(Xt)
    plt.figure(figsize=(7,5))
    if labels is None:
        plt.scatter(Z[:,0], Z[:,1], s=6, alpha=0.5)
    else:
        plt.scatter(Z[:,0], Z[:,1], s=6, c=labels, cmap="tab10", alpha=0.7)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()
