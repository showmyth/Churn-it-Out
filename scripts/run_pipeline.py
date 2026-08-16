from pathlib import Path

from src.data.load_data import load_data
from src.features.pipeline import clean_features


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "Telco-Churn.csv"

    print(f"Loading raw dataset: {raw_csv}")
    df = load_data(str(raw_csv))
    X, y = clean_features(df, target="Churn")

    print(f"Prepared features: {X.shape}")
    print(f"Prepared target: {y.shape}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
