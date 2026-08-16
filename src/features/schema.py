from dataclasses import dataclass, field

@dataclass
class RawData:
    columns: dict[str, type] = field(default_factory=lambda: {
        "customerID":       str,
        "gender":           str,
        "SeniorCitizen":    int,
        "Partner":          str,
        "Dependents":       str,
        "tenure":           int,
        "PhoneService":     str,
        "MultipleLines":    str,
        "InternetService":  str,
        "OnlineSecurity":   str,
        "OnlineBackup":     str,
        "DeviceProtection": str,
        "TechSupport":      str,
        "StreamingTV":      str,
        "StreamingMovies":  str,
        "Contract":         str,
        "PaperlessBilling": str,
        "PaymentMethod":    str,
        "MonthlyCharges":   float,
        "TotalCharges":     str,   # comes in as str, cast later
        "Churn":            str,
    })
    nullable_columns: list[str] = field(default_factory=lambda: [
        "TotalCharges",
        "MultipleLines",
    ])
    drop_columns: list[str] = field(default_factory=lambda: ["customerID"])
    binary_columns: dict[str, dict] = field(default_factory=lambda: {
        "gender":           {"Male": 1, "Female": 0},
        "Partner":          {"Yes": 1, "No": 0},
        "Dependents":       {"Yes": 1, "No": 0},
        "PhoneService":     {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "Churn":            {"Yes": 1, "No": 0},
    })
    ohe_columns: list[str] = field(default_factory=lambda: [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ])
    numerical_columns: list[str] = field(default_factory=lambda: [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ])

@dataclass
class ProcessedData:
    target_column: str = "Churn"

RAW_SCHEMA = RawData()
MODEL_INPUT_SCHEMA = ProcessedData()