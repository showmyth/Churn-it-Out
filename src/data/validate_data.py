from typing import List, Tuple

import pandas as pd
# pyright: reportPrivateImportUsage=false


def _load_great_expectations():
    try:
        import great_expectations as gx
        from great_expectations import expectations as gxe
    except ModuleNotFoundError as exc:
        if exc.name != "great_expectations":
            raise
        return None, None

    return gx, gxe


def _build_expectation_suite(gx, gxe):
    suite = gx.ExpectationSuite(name="telco_raw")

    for expectation in [
        gxe.ExpectColumnToExist(column="customerID"),
        gxe.ExpectColumnValuesToNotBeNull(column="customerID"),
        gxe.ExpectColumnToExist(column="gender"),
        gxe.ExpectColumnToExist(column="Partner"),
        gxe.ExpectColumnToExist(column="Dependents"),
        gxe.ExpectColumnToExist(column="PhoneService"),
        gxe.ExpectColumnToExist(column="InternetService"),
        gxe.ExpectColumnToExist(column="Contract"),
        gxe.ExpectColumnToExist(column="tenure"),
        gxe.ExpectColumnToExist(column="MonthlyCharges"),
        gxe.ExpectColumnToExist(column="TotalCharges"),
        gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]),
        gxe.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]),
        gxe.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]),
        gxe.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]),
        gxe.ExpectColumnValuesToBeInSet(
            column="Contract",
            value_set=["Month-to-month", "One year", "Two year"],
        ),
        gxe.ExpectColumnValuesToBeInSet(
            column="InternetService",
            value_set=["DSL", "Fiber optic", "No"],
        ),
        gxe.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]),
        gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120),
        gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200),
        gxe.ExpectColumnValuesToNotBeNull(column="tenure"),
        gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.98,
        ),
    ]:
        suite.add_expectation(expectation)

    return suite


def _failed_expectation_name(result) -> str:
    expectation_config = getattr(result, "expectation_config", None)
    if expectation_config is None and isinstance(result, dict):
        expectation_config = result.get("expectation_config", {})

    expectation_type = getattr(expectation_config, "type", None)
    if expectation_type:
        return expectation_type

    if isinstance(expectation_config, dict):
        return expectation_config.get("type") or expectation_config.get(
            "expectation_type", "unknown_expectation"
        )

    return "unknown_expectation"


def _validate_data_with_pandas(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    checks = {
        "expect_column_to_exist": [
            "customerID",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "InternetService",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],
        "expect_column_values_to_not_be_null": [
            "customerID",
            "tenure",
            "MonthlyCharges",
        ],
    }
    failures = []

    for column in checks["expect_column_to_exist"]:
        if column not in df.columns:
            failures.append(f"expect_column_to_exist:{column}")

    for column in checks["expect_column_values_to_not_be_null"]:
        if column in df.columns and df[column].isna().any():
            failures.append(f"expect_column_values_to_not_be_null:{column}")

    allowed_values = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "Churn": {"Yes", "No"},
    }
    for column, values in allowed_values.items():
        if column in df.columns:
            actual = set(df[column].dropna().astype(str).str.strip())
            if not actual.issubset(values):
                failures.append(f"expect_column_values_to_be_in_set:{column}")

    numeric_ranges = {
        "tenure": (0, 120),
        "MonthlyCharges": (0, 200),
    }
    for column, (minimum, maximum) in numeric_ranges.items():
        if column in df.columns and not df[column].between(minimum, maximum).all():
            failures.append(f"expect_column_values_to_be_between:{column}")

    if {"TotalCharges", "MonthlyCharges"}.issubset(df.columns):
        comparable = df[["TotalCharges", "MonthlyCharges"]].dropna()
        success_ratio = (comparable["TotalCharges"] >= comparable["MonthlyCharges"]).mean()
        if success_ratio < 0.98:
            failures.append("expect_column_pair_values_a_to_be_greater_than_b")

    return len(failures) == 0, failures


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the churn dataset using Great Expectations 1.x expectations."""
    gx, gxe = _load_great_expectations()

    df = df.copy()
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if gx is None or gxe is None:
        return _validate_data_with_pandas(df)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telco_pandas")
    data_asset = data_source.add_dataframe_asset(name="telco_raw")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("raw_dataframe")

    suite = _build_expectation_suite(gx, gxe)
    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        data=batch_definition,
        name="telco_raw_validation",
        suite=suite,
    )

    context.validation_definitions.add(validation_definition)  # ← add this
 
    result = validation_definition.run(batch_parameters={"dataframe": df})

    failed_tests = [
        _failed_expectation_name(item)
        for item in result.results
        if not getattr(item, "success", False)
    ]

    return bool(result.success), failed_tests
