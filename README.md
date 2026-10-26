# Credit Card Fraud Pattern Mining

Data Warehousing and Data Mining (DWDM) individual project — S Prathyumna, SRM IST Ramapuram.

Fraud is rare (well under 1% of card transactions), so it hides inside millions of ordinary
purchases. This project uses classic data mining techniques to find **what fraudulent
transactions have in common** and **which transactions look unusual**, without training a
supervised classifier:

| Stage | Technique | Output |
|---|---|---|
| Preprocessing | Deduplication, PII removal, feature engineering (hour, age, customer–merchant distance, log amount) | Clean feature table |
| Exploratory analysis | Fraud rate by category, hour and amount | `reports/figures/01–03` |
| Association rules | FP-Growth / Apriori on discretised baskets, rules `{…} → fraud=YES` ranked by lift | `reports/fraud_rules.csv` |
| Clustering | K-Means, k chosen by silhouette score | `reports/cluster_profiles.csv`, figures 04–05 |
| Outlier detection | Isolation Forest (label not used for training) | figure 06 |
| Evaluation | Silhouette, Davies–Bouldin, ROC-AUC / PR-AUC and recall@top-k of the outlier score against the fraud label | `reports/results.json`, `reports/summary.md` |

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(Kaggle, simulated with the Sparkov generator): `fraudTrain.csv` 1,296,675 rows and
`fraudTest.csv` 555,719 rows, 23 columns. See [`data/README.md`](data/README.md) for download steps.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full run on the Kaggle data (place fraudTrain.csv in data/ first)
python -m src.main

# quick check on a small synthetic file with the same schema
python scripts/make_sample_data.py
python -m src.main --data data/sample_transactions.csv

# use Apriori instead of FP-Growth (same itemsets, slower)
python -m src.main --algorithm apriori
```

Thresholds (minimum support, confidence, lift, k range, sample sizes) live in `src/config.py`.

## Project layout

```
src/
  config.py        settings and thresholds
  preprocess.py    loading, profiling, cleaning, feature engineering, discretisation
  eda.py           exploratory charts
  association.py   FP-Growth / Apriori rule mining
  clustering.py    K-Means + silhouette / Davies–Bouldin
  outliers.py      Isolation Forest + evaluation against the fraud label
  main.py          runs the whole pipeline
scripts/
  make_sample_data.py   synthetic test data
docs/
  project_diary.docx    course project diary
reports/              generated results (git-ignored)
```

## Notes and limitations

- The dataset is simulated, so patterns describe the generator as much as real-world fraud.
- Support and confidence are computed on the full, highly imbalanced data; lift is the most
  useful measure for spotting fraud-linked combinations.
- K-Means and Isolation Forest are fitted on random samples (sizes in `config.py`) to keep
  run time reasonable on a laptop.
- Hour of day is treated as a linear number, so 23:00 and 00:00 look far apart.

## Planned improvements

- Cyclical (sin/cos) encoding for hour and weekday
- Per-card behavioural features (spend velocity, deviation from the card's usual amount)
- DBSCAN / Local Outlier Factor alongside Isolation Forest
- Validate patterns on `fraudTest.csv`
- A small Streamlit dashboard to browse rules and clusters
