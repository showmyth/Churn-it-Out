import sqlite3
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost as mlflow_xgboost

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")


def get_logged_model_artifact_path(run_id: str) -> str | None:
    db_path = ROOT_DIR / "mlflow.db"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT artifact_location
            FROM logged_models
            WHERE source_run_id = ?
            ORDER BY creation_timestamp_ms DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    return row[0] if row else None


def try_load_model_from_path(model_path: str):
    path = Path(model_path)
    if not path.exists():
        print(f"Model artifact directory does not exist on disk: {path}")
        return None

    if (path / "MLmodel").exists() or (path / "model.ubj").exists():
        print(f"Loading MLflow model from local path: {path}")
        return mlflow_xgboost.load_model(str(path))

    print(f"No MLflow model files found under: {path}")
    return None


runs = mlflow.search_runs(
    search_all_experiments=True,
    order_by=["metrics.recall DESC"],
)

valid_runs = runs[runs["metrics.recall"].notna()].copy()  # type: ignore
if valid_runs.empty:
    raise RuntimeError("No MLflow runs with a recorded recall metric were found.")

best_run = valid_runs.iloc[0]
run_id = best_run["run_id"]
model_name = best_run.get("tags.model_name") or best_run.get("tags.mlflow.runName")

print(f"Best recall run ID: {run_id}")
print(f"Model name: {model_name}")
print(f"Recall: {best_run['metrics.recall']}")

best_model = None

artifact_path = get_logged_model_artifact_path(run_id)
if artifact_path:
    print(f"Using artifact path from mlflow.db: {artifact_path}")
    best_model = try_load_model_from_path(artifact_path)
    if best_model is not None:
        print("Loaded model successfully:", type(best_model).__name__)
    else:
        print("No runnable model artifact was found for this best-recall run.")
else:
    candidate_uris = [
        f"runs:/{run_id}/model",
        f"runs:/{run_id}/{model_name}_model",
        f"models:/{model_name}/Production",
    ]

    last_error = None
    for model_uri in candidate_uris:
        try:
            print(f"Attempting to load model from: {model_uri}")
            best_model = mlflow.xgboost.load_model(model_uri)  # type: ignore
            print("Loaded model successfully:", type(best_model).__name__)
            break
        except Exception as exc:  # pragma: no cover - runtime MLflow lookup path
            last_error = exc
    else:
        print("No runnable model artifact was found for the best recall run.")
        if last_error is not None:
            print(f"Last MLflow load error: {last_error}")

# saving to serving dir
SERVING_DIR = ROOT_DIR / "src" / "serving"
print(f"Adding serving model to {SERVING_DIR}...")

try:
    mlflow_xgboost.save_model(
        xgb_model=best_model,
        path=str(SERVING_DIR) + "/model"
    )
except:
    raise Exception("Wow this did not work")



