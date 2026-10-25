"""Load the flood dataset and print a quick summary."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
TARGET = "FloodProbability"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download train.csv from Kaggle and put it in the data/ folder."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nInfo:")
    df.info()
    print("\nSummary statistics:")
    print(df.describe().T)
    print("\nMissing values:", int(df.isna().sum().sum()))
