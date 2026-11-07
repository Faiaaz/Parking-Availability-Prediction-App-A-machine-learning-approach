"""Clean Birmingham parking occupancy and engineer features with real labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BIRMINGHAM_CSV,
    DATA_PROCESSED,
    FEATURES_PATH,
    OCCUPANCY_FULL_THRESHOLD,
)


def load_birmingham(path: Path = BIRMINGHAM_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    df = df.rename(
        columns={
            "SystemCodeNumber": "parking_zone",
            "Capacity": "capacity",
            "Occupancy": "occupancy",
            "LastUpdated": "date_time",
        }
    )
    # Sensor noise: clip occupancy into [0, capacity].
    df["occupancy"] = df["occupancy"].clip(lower=0)
    df["occupancy"] = np.minimum(df["occupancy"], df["capacity"])
    df["occupancy_rate"] = df["occupancy"] / df["capacity"]
    # Real label: 1 = space available, 0 = nearly full / full.
    df["available"] = (df["occupancy_rate"] < OCCUPANCY_FULL_THRESHOLD).astype(int)
    return df.sort_values(["parking_zone", "date_time"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["date_time"].dt.hour
    out["minute"] = out["date_time"].dt.minute
    out["dayofweek"] = out["date_time"].dt.dayofweek
    out["month"] = out["date_time"].dt.month
    out["day"] = out["date_time"].dt.day
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    # Cyclical encodings keep 23:00 close to 00:00.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["time_of_day"] = out["hour"] + out["minute"] / 60.0
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone lags from prior readings only (no future / label leakage)."""
    out = df.copy()
    g = out.groupby("parking_zone", group_keys=False)
    out["occ_rate_lag1"] = g["occupancy_rate"].shift(1)
    out["occ_rate_lag2"] = g["occupancy_rate"].shift(2)
    out["occ_rate_roll3"] = g["occupancy_rate"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    out["occ_rate_delta_lag"] = out["occ_rate_lag1"] - out["occ_rate_lag2"]
    # Drop rows without history so lags are defined.
    out = out.dropna(subset=["occ_rate_lag1", "occ_rate_lag2"]).reset_index(drop=True)
    return out


# Deliberately excludes current occupancy — that would leak the label.
CALENDAR_FEATURES = [
    "capacity",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "time_of_day",
]
LAG_FEATURES = [
    "occ_rate_lag1",
    "occ_rate_lag2",
    "occ_rate_roll3",
    "occ_rate_delta_lag",
]
FEATURE_COLUMNS = CALENDAR_FEATURES + LAG_FEATURES

TARGET_CLASS = "available"
TARGET_REG = "occupancy_rate"
META_COLUMNS = ["date_time", "parking_zone", "occupancy", "capacity"]


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_birmingham()
    feat = add_calendar_features(df)
    feat = add_lag_features(feat)
    return feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Birmingham parking features")
    parser.add_argument("--input", type=Path, default=BIRMINGHAM_CSV)
    parser.add_argument("--output", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = load_birmingham(args.input)
    features = build_features(raw)
    # Prefer parquet; fall back to CSV if pyarrow/fastparquet missing.
    try:
        features.to_parquet(args.output, index=False)
        out_path = args.output
    except Exception:
        out_path = args.output.with_suffix(".csv")
        features.to_csv(out_path, index=False)

    n = len(features)
    avail_rate = features[TARGET_CLASS].mean()
    print(f"Wrote {n:,} rows → {out_path}")
    print(f"Zones: {features['parking_zone'].nunique()}")
    print(f"Date range: {features['date_time'].min()} → {features['date_time'].max()}")
    print(f"Available rate (<{OCCUPANCY_FULL_THRESHOLD:.0%} full): {avail_rate:.3f}")
    print(f"Nearly-full rate: {1 - avail_rate:.3f}")


if __name__ == "__main__":
    main()
