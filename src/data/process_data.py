import pandas as pd

from features.schema import RAW_SCHEMA


def _binary_mapping(s: pd.Series) -> pd.Series:
    """Map binary categorical columns to 0/1 while preserving missing values."""
    series = s.astype(str).str.strip()
    unique_values = set(series.dropna().unique())

    if unique_values == {"Yes", "No"}:
        return series.map({"No": 0, "Yes": 1}).astype("Int64")
    if unique_values == {"Male", "Female"}:
        return series.map({"Male": 1, "Female": 0}).astype("Int64")
    if unique_values == {"No", "Yes"}:
        return series.map({"No": 0, "Yes": 1}).astype("Int64")
    if unique_values == {"Female", "Male"}:
        return series.map({"Male": 1, "Female": 0}).astype("Int64")

    return s


def process_data(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Clean, encode, and normalize the raw churn dataset for model training."""
    df = df.copy()

    # 1. Clean headers and drop redundant columns
    df.columns = [str(col).strip() for col in df.columns]

    drop_cols = [col for col in RAW_SCHEMA.drop_columns if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # 2. Encode target column
    if target in df.columns and pd.api.types.is_object_dtype(df[target]):
        encoded_target = df[target].astype(str).str.strip()
        df[target] = encoded_target.map({"Yes": 1, "No": 0}).astype("Int64")

    # 3. Convert numeric-like columns to numeric
    for col in RAW_SCHEMA.nullable_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Fill missing numeric values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # 5. Encode binary and multi-category columns
    obj_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != target]
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    binary_cols = [col for col in obj_cols if df[col].dropna().nunique() == 2]
    multi_cols = [col for col in obj_cols if df[col].dropna().nunique() > 2]

    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    for col in binary_cols:
        df[col] = _binary_mapping(df[col])
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0).astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df

