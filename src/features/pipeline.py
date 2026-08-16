from data import load_data, validate_data, process_data
from schema import RAW_SCHEMA, MODEL_INPUT_SCHEMA
import pandas as pd
from pathlib import Path
from typing import Tuple

# pipeline.py
PIPELINE_DIR = Path(__file__).parent          # src/features/
SRC_DIR      = PIPELINE_DIR.parent            # src/
ROOT_DIR     = SRC_DIR.parent                 # .
RAW_DATA     = ROOT_DIR / "data" / "raw" / "raw.csv" #./data/raw/raw.csv -> target location

def validate_against_schema(df: pd.DataFrame) -> bool:
    """Validate raw dataframe against schema using Great Expectations"""
    is_valid, failed_tests = validate_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed. Failed tests: {failed_tests}")
    return is_valid

def clean_features(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: validate → process → split into (X, y)
    
    Processing includes:
    - Header cleaning and dropping redundant columns
    - Encoding target and binary features
    - One-hot encoding multi-category features
    - Filling missing values
    
    Args:
        df: Raw dataframe
        target: Target column name
    
    Returns:
        (X, y): Features and target, ready for model
    """
    # 1. Validate raw data
    print("Step 1: Validating raw data against schema...")
    validate_against_schema(df)
    
    # 2. Process and feature-engineer data
    print("Step 2: Processing and engineering features...")
    processed_df = process_data(df, target=target)
    
    # 3. Split X and y
    print("Step 3: Splitting features and target...")
    if target not in processed_df.columns:
        raise ValueError(f"Target column '{target}' not found in processed data")
    
    y = processed_df[target]
    X = processed_df.drop(target, axis=1)
    
    # 4. Validate output shape
    print("Step 4: Validating output...")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y



















def pipeline_from_file(file_path: str, target: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """
    End-to-end pipeline: Load CSV → Validate → Process → Return (X, y)
    
    Args:
        file_path: Path to raw CSV
        target: Target column name
    
    Returns:
        (X, y): Features and target, ready for model
    """
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    return clean_features(df, target=target)
