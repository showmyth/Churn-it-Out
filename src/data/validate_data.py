import great_expectations as ge 
import pandas as pd
from typing import Tuple, List

def validate_data(df : pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Data validation for given Customer Churn dataset using Great Expectations.
        
    Implements data quality checks that must pass before model training.
     
    Validates data integrity, business logic constraints, and statistical properties
    that the ML model expects. 
    
    Args:
        df (pd.DataFrame): Input Dataframe

    Returns:
        tuple: results["success"] / results["failure"]
    """

    print("Validating Data...")

    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df) # type: ignore

    # === Schema Validation ===
    print("1. Validating Schema...")

    # 1. Customer Identifier must exist
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")

    # 2. Must contain the essential demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    # 3. Must contain service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    # 4. Key Predictors - Financial features
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    # === Business Logic ===
    print("2. Validating Business Logic Requirements...")

    # 1. Check for faults in gender assignment (in this case, must be M/F)
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    # 2. Yes/No fields should be filled appropriately
    yes_or_no_cols = ["Partner", "Dependants", "PhoneService"]

    for col in yes_or_no_cols:
        ge_df.expect_column_values_to_be_in_set(col, ["Yes", "No"])

    # 3. Validate Contract Types
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])

    # 4. Validate Internet Service Types
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    # === Numeric Range Validation ===
    print("3. Validating Value Ranges...")

    # 1. Tenure must not be negative
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)

    # 1.5 Tenure time should not be outrageous max <= 120 mos / 10 yrs
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

    # 2. Monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    # 3. Ensure against missing values in critical numeric features 
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    # === Data Consistency ===
    print("4. Validating data consistency...")

    # 1. Total charges should generally be >= Monthly charges (except for very new customers)
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.98  # Allow 2% exceptions for edge cases
    )

    # === Run the Test ===
    print("Running the complete validation...")
    res = ge_df.validate()

    # === PROCESS RESULTS ===
    failed_tests = []
    for r in res["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_tests.append(expectation_type)

    # Print validation summary
    total_checks = len(res["results"])
    passed_checks = sum(1 for r in res["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if res["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")

    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed tests: {failed_tests}")

    return res["success"], failed_tests

# EOF