import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import mlflow.xgboost as mlflow_xgboost


def train_model(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    df: pd.DataFrame | None = None,
    target: str | None = None,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    threshold: float = 0.5,
    log_to_mlflow: bool = True,
    model_name: str = "model",
    nested: bool = False,
) -> tuple[XGBClassifier, dict]:
    """Train an XGBoost classifier from either a feature matrix and target series or a data frame with a target column."""
    if df is not None and target is not None:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        X = df.drop(columns=[target])
        y = df[target]
    elif X is not None and y is not None:
        X = X.copy()
        y = y.copy()
    else:
        raise ValueError("Provide either (df, target) or (X, y).")

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        X_train = X
        y_train = y

    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test are required after train/test split.")

    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    preds = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))

    if log_to_mlflow:
        with mlflow.start_run(run_name=model_name, nested=nested):
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("threshold", threshold)
            mlflow.log_params(default_params)
            mlflow.log_metrics(metrics)
            mlflow_xgboost.log_model(model, f"{model_name}_model")

            try:
                train_ds = mlflow.data.from_pandas(pd.concat([X, y], axis=1), source="training_data")  # type: ignore
            except AttributeError:
                from mlflow.data.pandas_dataset import from_pandas as pd_from_pandas
                train_ds = pd_from_pandas(pd.concat([X, y], axis=1), source="training_data")

            mlflow.log_input(train_ds, context="training")

    print(
        f"{model_name} trained. "
        f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}"
    )
    return model, metrics