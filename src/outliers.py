"""Unsupervised outlier detection with Isolation Forest.

The fraud label is NOT used to train the model; it is only used afterwards
to check how many of the flagged outliers are actually fraud.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config


def run(df: pd.DataFrame) -> dict:
    sample = df.sample(min(config.IFOREST_SAMPLE, len(df)), random_state=config.RANDOM_STATE)
    X = StandardScaler().fit_transform(sample[config.NUMERIC_FEATURES])
    y = sample[config.TARGET].to_numpy()

    model = IsolationForest(n_estimators=200, contamination=config.IFOREST_CONTAMINATION,
                            random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X)
    score = -model.score_samples(X)          # higher = more anomalous
    flagged = model.predict(X) == -1

    base_rate = y.mean()
    result = {
        "rows_scored": int(len(sample)),
        "outliers_flagged": int(flagged.sum()),
        "outlier_pct": round(100 * flagged.mean(), 2),
        "fraud_rate_overall_pct": round(100 * base_rate, 3),
        "fraud_rate_in_outliers_pct": round(100 * y[flagged].mean(), 3) if flagged.any() else 0.0,
        "roc_auc_vs_label": round(float(roc_auc_score(y, score)), 4),
        "pr_auc_vs_label": round(float(average_precision_score(y, score)), 4),
    }
    order = np.argsort(-score)
    for pct in (1, 5):
        top = order[: max(1, int(len(order) * pct / 100))]
        result[f"recall_top{pct}pct"] = round(float(y[top].sum() / max(1, y.sum())), 4)
        result[f"precision_top{pct}pct"] = round(float(y[top].mean()), 4)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(score[y == 0], bins=60, density=True, alpha=0.6, color="#4C78A8", label="Genuine")
    ax.hist(score[y == 1], bins=60, density=True, alpha=0.6, color="#E45756", label="Fraud")
    ax.set_xlabel("Isolation Forest anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly scores: fraud vs genuine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "06_isolation_forest_scores.png", dpi=150)
    plt.close(fig)
    return result
