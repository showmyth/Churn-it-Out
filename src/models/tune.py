import mlflow
import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def tune_model(
    X,
    y,
    *,
    n_trials: int = 30,
    cv: int = 3,
    threshold: float = 0.5,
    model_name: str = "hypertuned",
    nested: bool = False,
) -> dict:
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        recall_scores = []
        for train_idx, valid_idx in folds.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_valid)[:, 1]
            preds = (probabilities >= threshold).astype(int)
            recall_scores.append(recall_score(y_valid, preds, pos_label=1, zero_division=0))

        return sum(recall_scores) / len(recall_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Params:", study.best_params)
    print("Best Recall:", study.best_value)

    with mlflow.start_run(run_name=model_name, nested=nested):
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_params({
            "tune_n_trials": n_trials,
            "tune_cv": cv,
            "tune_threshold": threshold,
            "tune_scale_pos_weight": scale_pos_weight,
        })
        mlflow.log_metric("recall", float(study.best_value))
        mlflow.log_params(study.best_params)

    return study.best_params