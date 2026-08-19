import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import pandas as pd
import mlflow.xgboost as mlflow_xgboost
from src.features.pipeline import clean_features



# Defining Model Paths   
SERVING_DIR = ROOT_DIR / "src" / "serving"
MODEL_PATH = SERVING_DIR / "model"

xgb_model = mlflow_xgboost.load_model(str(MODEL_PATH))

print("The XGBoost Model is:", xgb_model)


def predict(raw_input: dict) -> dict:
    df = pd.DataFrame([raw_input])
    X, _ = clean_features(df, validate=False)
    probs = xgb_model.predict_proba(X)[0][1]
    label = int(probs >= 0.2)

    return {"churn": label, "probability": round(float(probs), 4)}


