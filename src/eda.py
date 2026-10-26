"""Exploratory analysis: fraud rate by category / hour / amount."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

GENUINE = "#4C78A8"
FRAUD = "#E45756"


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    t = config.TARGET
    by_cat = df.groupby("category")[t].mean().mul(100).sort_values()
    by_hour = df.groupby("hour")[t].mean().mul(100)
    amt = df.groupby(t)["amt"].describe()[["mean", "50%", "max"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_cat.plot.barh(ax=ax, color=FRAUD)
    ax.set_xlabel("Fraud rate (%)")
    ax.set_ylabel("")
    ax.set_title("Fraud rate by merchant category")
    _save(fig, "01_fraud_rate_by_category.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    by_hour.plot(ax=ax, marker="o", color=FRAUD)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Fraud rate by hour of transaction")
    ax.grid(alpha=0.3)
    _save(fig, "02_fraud_rate_by_hour.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, name in [(0, GENUINE, "Genuine"), (1, FRAUD, "Fraud")]:
        ax.hist(df.loc[df[t] == label, "log_amt"], bins=60, density=True,
                alpha=0.6, color=color, label=name)
    ax.set_xlabel("log(1 + amount)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount: fraud vs genuine")
    ax.legend()
    _save(fig, "03_amount_distribution.png")

    return {
        "top_fraud_categories_pct": by_cat.sort_values(ascending=False).head(5).round(3).to_dict(),
        "peak_fraud_hours_pct": by_hour.sort_values(ascending=False).head(4).round(3).to_dict(),
        "amount_by_class": amt.round(2).rename(index={0: "genuine", 1: "fraud"}).to_dict(orient="index"),
    }
