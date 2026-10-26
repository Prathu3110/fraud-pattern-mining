"""Write a small SYNTHETIC file with the same schema as the Kaggle CSVs.

Only for checking that the pipeline runs without downloading 500 MB.
Numbers produced from it are not real findings.

    python scripts/make_sample_data.py --rows 50000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["grocery_pos", "shopping_net", "misc_net", "shopping_pos", "gas_transport",
              "home", "kids_pets", "entertainment", "food_dining", "personal_care",
              "health_fitness", "misc_pos", "travel", "grocery_net"]
RISKY = {"grocery_pos", "shopping_net", "misc_net"}


def make(rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud = rng.random(rows) < fraud_rate
    cat = rng.choice(CATEGORIES, rows)
    risky_cats = np.array(sorted(RISKY))
    cat[fraud] = np.where(rng.random(fraud.sum()) < 0.75,
                          rng.choice(risky_cats, fraud.sum()), cat[fraud])
    amt = np.round(np.exp(rng.normal(3.6, 1.1, rows)), 2)
    amt[fraud] = np.round(rng.uniform(200, 1200, fraud.sum()), 2)

    start = pd.Timestamp("2019-01-01")
    secs = rng.integers(0, 365 * 24 * 3600 * 1.5, rows)
    ts = start + pd.to_timedelta(secs, unit="s")
    hours = rng.choice([22, 23, 0, 1, 2, 3], fraud.sum())
    ts_f = ts[fraud].normalize() + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 3600, fraud.sum()), unit="s")
    ts = pd.Series(ts)
    ts[fraud] = ts_f

    lat = rng.uniform(25, 48, rows)
    lon = rng.uniform(-123, -70, rows)
    dob = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 50, rows), unit="D")

    return pd.DataFrame({
        "Unnamed: 0": np.arange(rows),
        "trans_date_trans_time": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": rng.integers(10**15, 10**16, rows),
        "merchant": ["fraud_Merchant " + str(i) for i in rng.integers(0, 700, rows)],
        "category": cat,
        "amt": amt,
        "first": "First", "last": "Last",
        "gender": rng.choice(["F", "M"], rows),
        "street": "1 Example St", "city": "City", "state": rng.choice(["NY", "TX", "CA", "PA"], rows),
        "zip": rng.integers(10000, 99999, rows),
        "lat": lat, "long": lon,
        "city_pop": rng.integers(100, 2_000_000, rows),
        "job": "Engineer",
        "dob": dob.strftime("%Y-%m-%d"),
        "trans_num": [f"{i:032x}" for i in range(rows)],
        "unix_time": (ts.astype("int64") // 10**9),
        "merch_lat": lat + rng.normal(0, 0.5, rows),
        "merch_long": lon + rng.normal(0, 0.5, rows),
        "is_fraud": fraud.astype(int),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--fraud-rate", type=float, default=0.006)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"
    make(args.rows, args.fraud_rate, args.seed).to_csv(out, index=False)
    print(f"Wrote {args.rows:,} synthetic rows to {out}")
