import pandas as pd
from features.schema import RAW_SCHEMA, MODEL_INPUT_SCHEMA


def _binary_mapping(s: pd.Series) -> pd.Series:
    """Map binary categorical columns to 0/1"""
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valSet = set(vals)

    if valSet == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    elif valSet == {"Male", "Female"}:
        return s.map({"Male": 1, "Female": 0}).astype("Int64")

    return s


def process_data(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """
    Complete data processing pipeline:
    1. Clean headers and drop redundant columns
    2. Encode target variable and binary features
    3. One-hot encode multi-category features
    4. Handle missing values and data types
    
    Args:
        df: Raw dataframe
        target: Target column name
    
    Returns:
        Processed dataframe ready for modeling
    """
    df = df.copy()
    
    # 1. Clean Headers
    df.columns = df.columns.str.strip()

    # 2. Drop Redundant Columns (e.g., IDs)
    drop_cols = RAW_SCHEMA.drop_columns
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col)

    # 3. Encode target column
    if target in df.columns and df[target].dtype == "object":
        df[target] = df[target].str.strip().map({"Yes": 1, "No": 0})

    # 4. Convert nullable columns to numeric
    for col in RAW_SCHEMA.nullable_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Fill missing values in numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # 6. Feature Engineering: Identify and encode categorical columns
    obj_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != target]
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # Split categorical columns by cardinality
    binary_cols = [col for col in obj_cols if df[col].dropna().nunique() == 2]
    multi_cols = [col for col in obj_cols if df[col].dropna().nunique() > 2]

    # 7. Convert boolean columns to 0/1
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # 8. Binary Encoding
    for col in binary_cols:
        df[col] = _binary_mapping(df[col].astype(str))
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0).astype(int)

    # 9. One-Hot Encoding for multi-category features
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df

