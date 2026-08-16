import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import eval, train, tune
import mlflow

DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "telco_churn_processed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate an XGBoost model from processed data."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processed CSV path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument("--target", default="Churn", help="Target column name.")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trial count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive-class threshold.")
    return parser.parse_args()


def load_processed_data(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")

    X, y = load_processed_data(args.input_csv, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Loaded processed data from {args.input_csv}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    with mlflow.start_run(run_name="pipeline"):
        print("\nTraining baseline model...")
        baseline_model, baseline_metrics = train(
            X_train, y_train, X_test=X_test, y_test=y_test,
            threshold=args.threshold, model_name="baseline", nested=True,
        )
        print("Baseline metrics:", baseline_metrics)

        print("\nBaseline holdout evaluation...")
        eval(baseline_model, X_test, y_test, threshold=args.threshold, model_name="baseline", nested=True)

        print("\nTuning model...")
        best_params = tune(
            X_train, y_train, n_trials=args.trials,
            threshold=args.threshold, model_name="hypertuned", nested=True,
        )
        best_params.update({
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 42, "n_jobs": -1, "eval_metric": "logloss",
        })

        print("\nTraining hypertuned model...")
        tuned_model, tuned_metrics = train(
            X_train, y_train, params=best_params, X_test=X_test, y_test=y_test,
            threshold=args.threshold, model_name="hypertuned", nested=True,
        )
        print("Hypertuned metrics:", tuned_metrics)

        print("\nHypertuned holdout evaluation...")
        eval(tuned_model, X_test, y_test, threshold=args.threshold, model_name="hypertuned", nested=True)


if __name__ == "__main__":
    main()