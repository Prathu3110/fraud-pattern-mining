"""Loading, cleaning and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The Kaggle CSVs carry a leftover pandas index column.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df


def profile(df: pd.DataFrame) -> dict:
    """Numbers that go into the 'Dataset Important Parameters' table."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return {
        "records": int(len(df)),
        "attributes": int(df.shape[1]),
        "numerical_attributes": numeric,
        "categorical_attributes": categorical,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "fraud_count": int(df[config.TARGET].sum()),
        "fraud_rate_pct": round(100 * df[config.TARGET].mean(), 3),
        "date_min": str(df["trans_date_trans_time"].min()),
        "date_max": str(df["trans_date_trans_time"].max()),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = df.drop(columns=[c for c in config.PII_COLUMNS if c in df.columns])

    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.day_name()
    df["month"] = ts.dt.to_period("M").astype(str)
    df["age"] = ((ts - dob).dt.days / 365.25).round(1)
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["log_amt"] = np.log1p(df["amt"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)
    return df


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Turn each transaction into a basket of categorical items for Apriori."""
    out = pd.DataFrame(index=df.index)
    out["amt"] = pd.cut(
        df["amt"], [-np.inf, 20, 100, 300, 800, np.inf],
        labels=["<20", "20-100", "100-300", "300-800", ">800"],
    )
    out["time"] = pd.cut(
        df["hour"], [-1, 5, 11, 17, 21, 23],
        labels=["night(0-5)", "morning(6-11)", "afternoon(12-17)", "evening(18-21)", "late(22-23)"],
    )
    out["age"] = pd.cut(df["age"], [0, 25, 40, 60, 200], labels=["<25", "25-40", "40-60", "60+"])
    out["category"] = df["category"]
    out["gender"] = df["gender"]
    out["day"] = np.where(df["weekday"].isin(["Saturday", "Sunday"]), "weekend", "weekday")
    out["fraud"] = np.where(df[config.TARGET] == 1, "YES", "NO")
    return out.astype(str).apply(lambda col: col.name + "=" + col)
