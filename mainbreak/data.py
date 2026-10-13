"""Loading the mains table and splitting it the one way this project splits it."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mains.csv"

TARGET = "burst_next_12m"
NUMERIC_FEATURES = [
    "age_years",
    "diameter_mm",
    "length_m",
    "pressure_bar",
    "soil_corrosivity",
    "winter_min_c",
    "prior_breaks_5y",
]
CATEGORICAL_FEATURES = ["material", "soil_type", "traffic_load", "district"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    """Return (X, y). X holds the eleven feature columns, y the 0/1 target."""
    frame = pd.read_csv(path)
    return frame[FEATURES], frame[TARGET]


def split(X, y):
    """The fixed 75/25 split, stratified on the target."""
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def load_split(path=DATA_PATH):
    """Return X_train, X_test, y_train, y_test from the table on disk."""
    X, y = load_data(path)
    return split(X, y)


def data_sha256(path=DATA_PATH):
    """SHA-256 of the table, so a run can say which bytes it was trained on."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
