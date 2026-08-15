import os
import pandas as pd
import polars as pl

# great expectations does have polars integration (without using pyarrow/to_pandas())
# so it is advised to just use pandas instead.

# why are we still using polars? To push the propaganda.

def load_data(file_path : str) -> pd.DataFrame:
        if not os.path.exists(file_path):
                raise FileNotFoundError(f"This file path: {file_path} does NOT exist")

        return (pl.read_csv(file_path)).to_pandas()