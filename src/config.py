"""Project paths and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

BIRMINGHAM_CSV = DATA_RAW / "dataset.csv"
MELBOURNE_CSV = DATA_RAW / "melbourne_bay_sensors.csv"
FEATURES_PATH = DATA_PROCESSED / "birmingham_features.parquet"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Nearly-full threshold: occupancy at or above this share of capacity ⇒ unavailable.
OCCUPANCY_FULL_THRESHOLD = 0.85

# Chronological hold-out: last 20% of time stamps for test.
TEST_FRACTION = 0.20

RANDOM_STATE = 42
