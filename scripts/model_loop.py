import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
from src.models import eval, train, tune

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
    y = df[target]

    if y.dtype == object:
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0, "1": 1, "0": 0}
        y = y.map(mapping)
        if y.isna().any():
            raise ValueError(
                f"Target column '{target}' contains unexpected values. "
                f"Expected one of: {list(mapping.keys())}"
            )
    y = y.astype(int)
    return X, y


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")

    X, y = load_processed_data(args.input_csv, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Loaded processed data from {args.input_csv}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    with mlflow.start_run(run_name="pipeline"):

        # ── Baseline ──────────────────────────────────────────
        with mlflow.start_run(run_name="baseline", nested=True):
            print("\nTraining baseline model...")
            baseline_model, baseline_metrics = train(
                X_train, y_train,
                X_test=X_test, y_test=y_test,
                threshold=args.threshold,
                model_name="baseline",
                # log_to_mlflow=True (default): logs into this active nested run
            )
            print("Baseline metrics:", baseline_metrics)

            print("\nBaseline holdout evaluation...")
            eval(
                baseline_model, X_test, y_test,
                threshold=args.threshold,
                model_name="baseline_evals",
                nested=True,
            )

        # ── Hypertuned ────────────────────────────────────────
        with mlflow.start_run(run_name="hypertuned", nested=True):
            print("\nTuning model...")
            # tune() logs into this active nested run directly — no sub-run opened
            best_params, _ = tune(
                X_train, y_train,
                n_trials=args.trials,
                threshold=args.threshold,
                scale_pos_weight=scale_pos_weight,
                model_name="hypertuned",
            )

            print("\nTraining hypertuned model...")
            # train() also logs into this same active nested run
            tuned_model, tuned_metrics = train(
                X_train, y_train,
                params=best_params,
                X_test=X_test, y_test=y_test,
                threshold=args.threshold,
                model_name="hypertuned",
            )
            print("Hypertuned metrics:", tuned_metrics)

            print("\nHypertuned holdout evaluation...")
            eval(
                tuned_model, X_test, y_test,
                threshold=args.threshold,
                model_name="hypertuned_evals",
                nested=True,
            )


if __name__ == "__main__":
    main()