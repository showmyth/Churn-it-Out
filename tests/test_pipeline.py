import pandas as pd

from src.data.load_data import load_data
from src.data.process_data import process_data
from src.data.validate_data import validate_data
from src.features.pipeline import clean_features


def test_process_data_generates_model_ready_frame():
    df = load_data("data/raw/Telco-Churn.csv")

    processed = process_data(df)

    assert "customerID" not in processed.columns
    assert "Churn" in processed.columns
    assert processed["Churn"].isin([0, 1]).all()
    assert processed.shape[0] == df.shape[0]


def test_validate_data_passes_for_raw_dataset():
    df = load_data("data/raw/Telco-Churn.csv")

    is_valid, failures = validate_data(df)

    assert is_valid is True
    assert failures == []


def test_clean_features_returns_X_y():
    df = load_data("data/raw/Telco-Churn.csv")

    X, y = clean_features(df)

    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
