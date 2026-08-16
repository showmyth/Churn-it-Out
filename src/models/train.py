import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import mlflow.xgboost as mlflow_xgboost


def train_model(X: pd.DataFrame | None = None, y: pd.Series | None = None, *, df: pd.DataFrame | None = None, target: str | None = None) -> None:
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("recall", float(rec))
        mlflow_xgboost.log_model(model, "model")

        try:
            train_ds = mlflow.data.from_pandas(pd.concat([X, y], axis=1), source="training_data")  # type: ignore
        except AttributeError:
            from mlflow.data.pandas_dataset import from_pandas as pd_from_pandas
            train_ds = pd_from_pandas(pd.concat([X, y], axis=1), source="training_data")

        mlflow.log_input(train_ds, context="training")
        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")

