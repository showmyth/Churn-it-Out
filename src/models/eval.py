import inspect
import warnings

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    *,
    threshold: float = 0.2,
    model_name: str = "baseline_eval",
    nested: bool = True,
) -> dict:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        preds = (probabilities >= threshold).astype(int)
    else:
        probabilities = None
        preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, preds, labels=[0, 1])

    metrics: dict = {
        "eval_accuracy": float(accuracy_score(y_test, preds)),
        "eval_precision": float(precision_score(y_test, preds, zero_division=0)),
        "eval_recall": float(recall_score(y_test, preds, zero_division=0)),
        "eval_f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    if probabilities is not None:
        if len(np.unique(y_test)) > 1:
            metrics["eval_roc_auc"] = float(roc_auc_score(y_test, probabilities))
        else:
            warnings.warn(f"[{model_name}] y_test has only one class — eval_roc_auc skipped.")

    print("Classification Report:\n", classification_report(y_test, preds, zero_division=0))
    print("Confusion Matrix:\n", matrix)

    # Convert matrix to plain Python ints so callers can serialise it freely
    tn, fp, fn, tp = int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0]), int(matrix[1][1])

    start_run_kwargs = {"run_name": f"{model_name}_evaluation"}
    try:
        if "nested" in inspect.signature(mlflow.start_run).parameters:
            start_run_kwargs["nested"] = nested  # type: ignore
    except (TypeError, ValueError):
        pass

    with mlflow.start_run(**start_run_kwargs):  # type: ignore
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("eval_threshold", threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({
            "eval_true_negative": tn,
            "eval_false_positive": fp,
            "eval_false_negative": fn,
            "eval_true_positive": tp,
        })
        mlflow.log_dict(report, "classification_report.json") # type: ignore

    return {
        "classification_report": report,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "metrics": metrics,
    }