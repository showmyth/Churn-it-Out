import pandas as pd
import mlflow
import mlflow.xgboost as mlflow_xgboost
from mlflow.data.pandas_dataset import from_pandas

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import Any


def train_model(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    df: pd.DataFrame | None = None,
    target: str | None = None,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    threshold: float = 0.2,
    log_to_mlflow: bool = True,
    log_dataset: bool = False,
    model_name: str = "model",
    nested: bool = True,
) -> tuple[XGBClassifier, dict]:
    """
    Train an XGBoost classifier.

    Accepts either (df, target) or (X, y).
    If X_test/y_test are not provided, splits internally (80/20 stratified).
    Dataset logging is off by default to avoid cost on repeated calls.
    """
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
        X_train, y_train = X, y

    default_params: dict = {
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
    if y_test is not None and preds is not None:
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
    else:
        acc, prec, rec, f1 = 0, 0, 0, 0
        
    metrics: dict = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    if y_test is not None and y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        import warnings
        warnings.warn(
            f"[{model_name}] y_test has only one class — roc_auc skipped. Check your split."
        )

    if log_to_mlflow:
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("has_model", "true")
        mlflow.log_param("threshold", threshold)
        mlflow.log_params(default_params)
        mlflow.log_metrics(metrics)
        mlflow_xgboost.log_model(model, name="model")

        if log_dataset:
            try:
                train_ds = from_pandas(
                    pd.concat([X_train, y_train], axis=1),
                    source="training_data",
                )
            except AttributeError:
                from mlflow.data.pandas_dataset import from_pandas as pd_from_pandas
                train_ds = pd_from_pandas(
                    pd.concat([X_train, y_train], axis=1),
                    source="training_data",
                )
            mlflow.log_input(train_ds, context="training")

    print(
        f"{model_name} trained. "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )
    return model, metrics