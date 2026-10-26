"""K-Means clustering of transactions on behavioural features."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


def run(df: pd.DataFrame) -> dict:
    sample = df.sample(min(config.CLUSTER_SAMPLE, len(df)), random_state=config.RANDOM_STATE)
    X = StandardScaler().fit_transform(sample[config.NUMERIC_FEATURES])
    score_idx = sample.index.get_indexer(
        sample.sample(min(config.SILHOUETTE_SAMPLE, len(sample)), random_state=1).index
    )

    scores = {}
    for k in config.K_RANGE:
        labels = KMeans(n_clusters=k, n_init=5, random_state=config.RANDOM_STATE).fit_predict(X)
        scores[k] = float(silhouette_score(X[score_idx], labels[score_idx]))
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=config.RANDOM_STATE)
    sample = sample.assign(cluster=km.fit_predict(X))

    profile = (
        sample.groupby("cluster")
        .agg(size=("amt", "size"), mean_amt=("amt", "mean"), mean_hour=("hour", "mean"),
             mean_age=("age", "mean"), mean_distance_km=("distance_km", "mean"),
             fraud_rate_pct=(config.TARGET, lambda s: 100 * s.mean()))
        .round(2)
    )
    profile.to_csv(config.REPORTS_DIR / "cluster_profiles.csv")

    # Elbow-style plot of silhouette vs k
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(scores), list(scores.values()), marker="o", color="#4C78A8")
    ax.axvline(best_k, ls="--", color="grey")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k for K-Means")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "04_kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    # 2-D PCA view of the clusters
    pts = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X[score_idx])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=sample["cluster"].values[score_idx], s=4, cmap="tab10")
    ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K-Means clusters (k={best_k}) in PCA space")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "05_kmeans_pca.png", dpi=150)
    plt.close(fig)

    return {
        "silhouette_by_k": {int(k): round(v, 4) for k, v in scores.items()},
        "best_k": int(best_k),
        "silhouette": round(scores[best_k], 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, sample["cluster"])), 4),
        "profiles": profile.reset_index().to_dict(orient="records"),
    }
