import mlflow
import optuna
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    X,
    y,
    *,
    n_trials: int = 100,
    cv: int = 3,
    threshold: float = 0.2,
    scale_pos_weight: float | None = None,
    model_name: str = "hypertuned",
    nested: bool = True,
) -> tuple[dict, float]:
    """
    Returns (best_params, scale_pos_weight).
    best_params is a complete dict ready to pass to train_model.
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    if scale_pos_weight is None:
        scale_pos_weight = float((y == 0).sum() / (y == 1).sum())

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
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight, scale_pos_weight * 3),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        recall_scores = []
        accuracy_scores = []
        for train_idx, valid_idx in folds.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= threshold).astype(int)
            recall_scores.append(recall_score(y_val, preds, pos_label=1, zero_division=0))
            accuracy_scores.append(accuracy_score(y_val, preds))

        trial.set_user_attr("cv_recall", sum(recall_scores) / len(recall_scores))
        trial.set_user_attr("cv_accuracy", sum(accuracy_scores) / len(accuracy_scores))
        return sum(recall_scores) / len(recall_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    best_trial = study.best_trial
    best_recall = float(best_trial.user_attrs.get("cv_recall", study.best_value))
    best_accuracy = float(best_trial.user_attrs.get("cv_accuracy", best_recall))

    print("Best params:", best_params)
    print("Best CV Recall:", best_recall)
    print("Best CV Accuracy:", best_accuracy)

    # Logged under the active run opened by pipeline.py — no new run opened here.
    # Metrics prefixed with "cv_" to avoid colliding with holdout "recall" from train_model,
    # which is what the loader filters on via tags.has_model = 'true'.
    mlflow.set_tag("model_name", model_name)
    mlflow.log_params({
        "tune_n_trials": n_trials,
        "tune_cv": cv,
        "tune_threshold": threshold,
        "tune_scale_pos_weight": scale_pos_weight,
    })
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_recall", best_recall)
    mlflow.log_metric("cv_accuracy", best_accuracy)

    return best_params, scale_pos_weight