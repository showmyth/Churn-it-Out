from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data.load_data import load_data
from src.data.process_data import process_data
from src.data.validate_data import validate_data

PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PIPELINE_DIR.parent
ROOT_DIR = SRC_DIR.parent
RAW_DATA = ROOT_DIR / "data" / "raw" / "Telco-Churn.csv"


def validate_against_schema(df: pd.DataFrame) -> bool:
    """Validate raw dataframe against the project's schema and business rules."""
    is_valid, failed_tests = validate_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed. Failed tests: {failed_tests}")
    return is_valid


def clean_features(df: pd.DataFrame, target: str = "Churn", validate: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate -> process -> split into features and target arrays."""
    if validate: 
        print("Validating raw data against schema...")
        validate_against_schema(df)

    print("Processing and engineering features...")
    processed_df = process_data(df, target=target)

    print("Splitting features and target...")
    if target not in processed_df.columns:
        raise ValueError(f"Target column '{target}' not found in processed data")

    y = processed_df[target]
    X = processed_df.drop(columns=[target])

    print("Validating output...")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    return X, y


def pipeline_from_file(file_path: str | Path, target: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV, validate it, process it, and return model-ready features and target."""
    file_path = Path(file_path)
    print(f"Loading data from {file_path}...")
    df = load_data(str(file_path))
    return clean_features(df, target=target)
