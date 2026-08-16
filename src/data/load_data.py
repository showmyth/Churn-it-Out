import os

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"This file path: {file_path} does NOT exist")

    return pd.read_csv(file_path)
