from typing import List, Tuple

import great_expectations as gx
from great_expectations import expectations as gxe
import pandas as pd
# pyright: reportPrivateImportUsage=false

def _build_expectation_suite() -> gx.ExpectationSuite:
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


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the churn dataset using Great Expectations 1.x expectations."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telco_pandas")
    data_asset = data_source.add_dataframe_asset(name="telco_raw")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("raw_dataframe")

    validation_definition = gx.ValidationDefinition(
        data=batch_definition,
        name="telco_raw_validation",
        suite=_build_expectation_suite(),
    )

    # create suite
    suite = _build_expectation_suite()
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
