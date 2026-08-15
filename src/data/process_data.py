import pandas as pd

def process_data(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """
    Cleaning/Processing Logic for a given Dataset
    - trims col names
    - drops Redundant cols
    - implements type casting on requirement
    - implements necessary encodings
    """

    # 1. Clean Headers
    df.columns = df.columns.str.strip()

    # 2. Drop Redundant Cols -> IDs in this case
    for col in ["customerId", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(col)

    # 3. Add Encodings
    if target in df.columns and df[target].dtype == "object":
        df[target] = df[target].str.strip().map({"Yes" : 1, "No": 0})

    # 4. Fill Missing Values

    # The Strategy -> 1. numeric: fill with zero 2. others: leave for(get_dummies ignores NaN safely)
    feature = "TotalCharges"
    if feature in df.columns:
        df[feature] = pd.to_numeric(df[feature], errors = "coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

