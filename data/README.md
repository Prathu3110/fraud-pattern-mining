# Data

This project uses the **Credit Card Transactions Fraud Detection Dataset** (Kaggle, by Kartik Shenoy),
a simulated dataset generated with the Sparkov data generator.

- Link: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Files: `fraudTrain.csv` (1,296,675 rows) and `fraudTest.csv` (555,719 rows), 23 columns each

Download the zip from Kaggle and place the CSVs in this folder:

```
data/fraudTrain.csv
data/fraudTest.csv
```

The CSVs are git-ignored (they are ~500 MB together). For a quick offline test of the pipeline,
`python scripts/make_sample_data.py` writes a small **synthetic** file with the same schema to
`data/sample_transactions.csv`. Results from that file are only for checking the code runs.
