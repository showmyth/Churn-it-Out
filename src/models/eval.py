import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    *,
    threshold: float = 0.5,
    model_name: str = "XGBoost",
    nested: bool = False,
) -> dict:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        preds = (probabilities >= threshold).astype(int)
    else:
        preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, preds, labels=[0, 1])
    metrics = {
        "eval_accuracy": float(accuracy_score(y_test, preds)),
        "eval_precision": float(precision_score(y_test, preds, zero_division=0)),
        "eval_recall": float(recall_score(y_test, preds, zero_division=0)),
        "eval_f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    print("Classification Report:\n", classification_report(y_test, preds, zero_division=0))
    print("Confusion Matrix:\n", matrix)

    with mlflow.start_run(run_name=f"{model_name}_evaluation", nested=nested):
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("eval_threshold", threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({
            "eval_true_negative": int(matrix[0][0]),
            "eval_false_positive": int(matrix[0][1]),
            "eval_false_negative": int(matrix[1][0]),
            "eval_true_positive": int(matrix[1][1]),
        })
        mlflow.log_dict(report, "classification_report.json")  # type: ignore

    return {
        "classification_report": report,
        "confusion_matrix": matrix,
        "metrics": metrics,
    }