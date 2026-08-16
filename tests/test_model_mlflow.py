from contextlib import nullcontext
from types import SimpleNamespace
import importlib

import pandas as pd


tune_module = importlib.import_module("src.models.tune")


def test_tune_logs_best_params_without_best_prefix(monkeypatch):
    logged = {}

    class DummyStudy:
        best_params = {"learning_rate": 0.05, "max_depth": 6}
        best_value = 0.81

        def optimize(self, *args, **kwargs):
            return None

    monkeypatch.setattr(tune_module.optuna, "create_study", lambda direction=None: DummyStudy())
    monkeypatch.setattr(tune_module.mlflow, "active_run", lambda: False)
    monkeypatch.setattr(
        tune_module.mlflow,
        "start_run",
        lambda run_name=None: nullcontext(),
    )
    monkeypatch.setattr(tune_module.mlflow, "set_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(tune_module.mlflow, "log_param", lambda *args, **kwargs: None)
    def capture_params(params):
        logged.setdefault("params_calls", []).append(params)
        logged["params"] = params

    monkeypatch.setattr(tune_module.mlflow, "log_params", capture_params)
    monkeypatch.setattr(tune_module.mlflow, "log_metric", lambda name, value: logged.setdefault("metrics", {}).update({name: value}))

    tune_module.tune_model(
        pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 0, 1, 0, 1, 0]}),
        pd.Series([0, 1, 0, 1, 0, 1]),
        n_trials=1,
        cv=2,
    )

    assert "learning_rate" in logged["params_calls"][-1]
    assert "max_depth" in logged["params_calls"][-1]
    assert "best_learning_rate" not in logged["params_calls"][-1]
    assert "best_max_depth" not in logged["params_calls"][-1]
    assert logged["metrics"]["recall"] == 0.81


def test_model_loop_uses_hypertuned_name(monkeypatch):
    from scripts import model_loop

    captured = []

    def fake_train(*args, **kwargs):
        captured.append(kwargs.get("model_name"))
        return object(), {"accuracy": 0.9, "recall": 0.8}

    monkeypatch.setattr(
        model_loop,
        "load_processed_data",
        lambda *args, **kwargs: (pd.DataFrame({"a": [0, 1, 0, 1]}), pd.Series([0, 1, 0, 1])),
    )
    monkeypatch.setattr(model_loop, "train", fake_train)
    monkeypatch.setattr(model_loop, "eval", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_loop, "tune", lambda *args, **kwargs: {"n_estimators": 10, "max_depth": 3})
    monkeypatch.setattr(model_loop.mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_loop,
        "parse_args",
        lambda: SimpleNamespace(input_csv="dummy.csv", target="Churn", trials=2, threshold=0.5),
    )
    monkeypatch.setattr(model_loop, "train_test_split", lambda X, y, **kwargs: (X, X, y, y))

    model_loop.main()

    assert captured == ["baseline", "hypertuned"]
