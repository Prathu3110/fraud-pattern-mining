"""Central settings for the fraud pattern mining pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

DEFAULT_TRAIN = DATA_DIR / "fraudTrain.csv"
DEFAULT_TEST = DATA_DIR / "fraudTest.csv"

RANDOM_STATE = 42
TARGET = "is_fraud"

# Columns that identify a person or a single transaction. They are dropped
# before analysis: they carry no general pattern and are personal data.
PII_COLUMNS = ["first", "last", "street", "cc_num", "trans_num", "zip"]

# Numeric features used for clustering and outlier detection.
NUMERIC_FEATURES = [
    "log_amt",
    "hour",
    "age",
    "distance_km",
    "log_city_pop",
]

# Association rule mining thresholds.
MIN_SUPPORT = 0.0005        # fraud is ~0.5% of rows, so support must be tiny
MIN_CONFIDENCE = 0.05
MIN_LIFT = 2.0
MAX_RULE_LEN = 4

# Clustering.
K_RANGE = range(3, 9)
CLUSTER_SAMPLE = 200_000    # rows used to fit K-Means
SILHOUETTE_SAMPLE = 10_000  # rows used to score silhouette

# Isolation Forest.
IFOREST_SAMPLE = 300_000
IFOREST_CONTAMINATION = 0.01   # flag the most anomalous 1%
