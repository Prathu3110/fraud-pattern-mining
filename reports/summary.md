# Results summary

Dataset: `/Volumes/DISKA/Dev/fraud-pattern-mining/data/fraudTrain.csv`  
Records: 1,296,675 | Attributes: 22 | Missing values: 0 | Duplicates: 0  
Fraud: 7,506 (0.579%) | Period: 2019-01-01 00:00:18 to 2020-06-21 12:13:37

## Top association rules ({...} -> fraud=YES)

| If | Support | Confidence | Lift |
|---|---|---|---|
| amt=300-800 AND category=grocery_pos | 0.0009 | 1.000 | 172.8 |
| amt=300-800 AND category=grocery_pos AND time=night(0-5) | 0.0006 | 1.000 | 172.8 |
| amt=300-800 AND category=grocery_pos AND day=weekday | 0.0006 | 1.000 | 172.8 |
| age=60+ AND amt=>800 AND time=late(22-23) | 0.0005 | 0.951 | 164.2 |
| amt=>800 AND category=shopping_net AND time=late(22-23) | 0.0010 | 0.898 | 155.0 |
| amt=>800 AND gender=M AND time=late(22-23) | 0.0010 | 0.881 | 152.1 |
| age=40-60 AND amt=>800 AND time=late(22-23) | 0.0005 | 0.779 | 134.6 |
| amt=>800 AND day=weekday AND time=late(22-23) | 0.0010 | 0.779 | 134.6 |
| amt=>800 AND time=late(22-23) | 0.0015 | 0.775 | 133.8 |
| amt=>800 AND category=shopping_net AND gender=M | 0.0008 | 0.687 | 118.7 |

## Clustering (K-Means)

Best k = 8 | silhouette = 0.1802 | Davies-Bouldin = 1.4555

| Cluster | Size | Mean amt | Mean hour | Mean age | Fraud rate % |
|---|---|---|---|---|---|
| 0 | 22500 | 27.17 | 17.65 | 36.85 | 0.11 |
| 1 | 19347 | 116.36 | 7.3 | 42.63 | 0.81 |
| 2 | 30968 | 97.78 | 4.39 | 43.67 | 0.81 |
| 3 | 26899 | 6.91 | 17.55 | 38.81 | 0.15 |
| 4 | 17243 | 6.29 | 6.44 | 64.97 | 0.32 |
| 5 | 31701 | 117.19 | 17.31 | 37.37 | 1.09 |
| 6 | 28172 | 77.68 | 15.7 | 39.92 | 0.47 |
| 7 | 23170 | 84.82 | 13.56 | 74.37 | 0.79 |

## Outlier detection (Isolation Forest)

Flagged 3,000 of 300,000 rows (1.0%).  
Fraud rate: 0.581% overall vs 8.367% among outliers.  
ROC-AUC vs label: 0.848 | PR-AUC: 0.0531 | Recall in top 1%: 0.1439 | top 5%: 0.3865
