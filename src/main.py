"""Run the full pipeline.

    python -m src.main                               # uses data/fraudTrain.csv
    python -m src.main --data data/sample_transactions.csv
    python -m src.main --algorithm apriori
"""
from __future__ import annotations

import argparse
import json
import time

from . import association, clustering, config, eda, outliers, preprocess


def write_summary(results: dict, rules) -> None:
    p, c, o = results["profile"], results["clustering"], results["outliers"]
    lines = [
        "# Results summary", "",
        f"Dataset: `{results['data']}`  ",
        f"Records: {p['records']:,} | Attributes: {p['attributes']} | "
        f"Missing values: {p['missing_values']} | Duplicates: {p['duplicate_records']}  ",
        f"Fraud: {p['fraud_count']:,} ({p['fraud_rate_pct']}%) | Period: {p['date_min']} to {p['date_max']}",
        "", "## Top association rules ({...} -> fraud=YES)", "",
        "| If | Support | Confidence | Lift |", "|---|---|---|---|",
    ]
    for _, r in rules.head(10).iterrows():
        lines.append(f"| {r['if']} | {r['support']:.4f} | {r['confidence']:.3f} | {r['lift']:.1f} |")
    lines += [
        "", "## Clustering (K-Means)", "",
        f"Best k = {c['best_k']} | silhouette = {c['silhouette']} | Davies-Bouldin = {c['davies_bouldin']}",
        "", "| Cluster | Size | Mean amt | Mean hour | Mean age | Fraud rate % |", "|---|---|---|---|---|---|",
    ]
    for r in c["profiles"]:
        lines.append(f"| {r['cluster']} | {r['size']} | {r['mean_amt']} | {r['mean_hour']} | "
                     f"{r['mean_age']} | {r['fraud_rate_pct']} |")
    lines += [
        "", "## Outlier detection (Isolation Forest)", "",
        f"Flagged {o['outliers_flagged']:,} of {o['rows_scored']:,} rows ({o['outlier_pct']}%).  ",
        f"Fraud rate: {o['fraud_rate_overall_pct']}% overall vs {o['fraud_rate_in_outliers_pct']}% among outliers.  ",
        f"ROC-AUC vs label: {o['roc_auc_vs_label']} | PR-AUC: {o['pr_auc_vs_label']} | "
        f"Recall in top 1%: {o['recall_top1pct']} | top 5%: {o['recall_top5pct']}",
    ]
    (config.REPORTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Credit card fraud pattern mining")
    ap.add_argument("--data", default=str(config.DEFAULT_TRAIN))
    ap.add_argument("--algorithm", choices=["fpgrowth", "apriori"], default="fpgrowth")
    args = ap.parse_args()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step(f"Loading {args.data}")
    raw = preprocess.load(args.data)
    results = {"data": args.data, "profile": preprocess.profile(raw)}

    step("Cleaning and feature engineering")
    df = preprocess.clean(raw)
    del raw

    step("Exploratory analysis")
    results["eda"] = eda.run(df)

    step(f"Association rule mining ({args.algorithm})")
    rules = association.run(preprocess.discretize(df), algorithm=args.algorithm)
    results["rules_found"] = int(len(rules))
    results["top_rules"] = rules.head(10).round(4).to_dict(orient="records")

    step("K-Means clustering")
    results["clustering"] = clustering.run(df)

    step("Isolation Forest outlier detection")
    results["outliers"] = outliers.run(df)

    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    write_summary(results, rules)
    step("Done. See reports/summary.md, reports/results.json and reports/figures/")


if __name__ == "__main__":
    main()
