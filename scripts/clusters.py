# scripts/cluster.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from src.utils.paths import RAW, MODELS, REPORTS
from src.features.preprocess import build_preprocessor, CHURN_LEAK_COLS, make_Xy, _sanitize

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main(args):
    df = pd.read_csv(RAW / "customer_churn.csv")

    # Derive numeric/categorical lists the same way as supervised
    X_all, y, num_cols, cat_cols = make_Xy(df, drop_noise=True)

    # --- NEW: sanitize BEFORE clustering (handles ' ', '', NA -> NaN and numeric coercion)
    df = _sanitize(df, numeric_cols=num_cols)

    # Drop churn-related columns for unsupervised features
    X_uns = df.drop(columns=[c for c in CHURN_LEAK_COLS if c in df.columns])

    pre = build_preprocessor(num_cols, cat_cols)
    Xt = pre.fit_transform(X_uns)
    # Pick k by silhouette (2..9)
    sil_scores = {}
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(Xt)
        sil_scores[k] = float(silhouette_score(Xt, labels))
    best_k = max(sil_scores, key=sil_scores.get)

    # Fit final KMeans
    km = KMeans(n_clusters=best_k, random_state=42)
    labels = km.fit_predict(Xt)

    # Save KMeans+preprocessor pipeline
    from sklearn.pipeline import Pipeline as SKLPipeline
    km_pipe = SKLPipeline([("pre", pre), ("kmeans", km)])
    dump(km_pipe, MODELS / f"kmeans_k{best_k}.joblib")

    # PCA for visualization (optional; saves the transformed 2D embedding)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xt)
    emb = pd.DataFrame({"pc1": Z[:,0], "pc2": Z[:,1], "cluster": labels})
    emb.to_csv(REPORTS / "unsupervised" / f"pca_k{best_k}.csv", index=False)

    # Attach clusters to the original df and save a compact profile + churn rates
    out = df.copy()
    out["Cluster"] = labels
    out.to_csv(REPORTS / "unsupervised" / f"clusters_k{best_k}.csv", index=False)

    # If true churn exists, evaluate ARI/NMI and churn rate by cluster
    metrics = {"best_k": best_k, "silhouette": sil_scores}
    if "Churn Value" in df.columns:
        ari = adjusted_rand_score(df["Churn Value"], labels)
        nmi = normalized_mutual_info_score(df["Churn Value"], labels)
        metrics.update({"ari": float(ari), "nmi": float(nmi)})

        churn_by_cluster = out.groupby("Cluster")["Churn Value"].mean().to_dict()
        metrics["churn_rate_by_cluster"] = {int(k): float(v) for k, v in churn_by_cluster.items()}

    save_json(metrics, REPORTS / "unsupervised" / f"kmeans_k{best_k}_metrics.json")

    # Also save a small numeric profile
    prof = out.groupby("Cluster")[["Tenure Months","Monthly Charges","Total Charges","CLTV"]].mean()
    prof.to_csv(REPORTS / "unsupervised" / f"kmeans_k{best_k}_numeric_profile.csv")

    print("\nArtifacts:")
    print(f"- KMeans pipeline: {MODELS / f'kmeans_k{best_k}.joblib'}")
    print(f"- PCA embedding:  {REPORTS / 'unsupervised' / f'pca_k{best_k}.csv'}")
    print(f"- Cluster labels: {REPORTS / 'unsupervised' / f'clusters_k{best_k}.csv'}")
    print(f"- Metrics:        {REPORTS / 'unsupervised' / f'kmeans_k{best_k}_metrics.json'}")
    print(f"- Profile:        {REPORTS / 'unsupervised' / f'kmeans_k{best_k}_numeric_profile.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised clustering for churn data.")
    args = parser.parse_args()
    main(args)
