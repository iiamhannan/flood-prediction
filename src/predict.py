"""Load the saved model and predict flood probability."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"


def load_model(path: Path = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_one(values: dict) -> float:
    """values: {feature_name: number}. Returns probability between 0 and 1."""
    model, features = load_model()
    row = pd.DataFrame([values])[features]
    return float(min(max(model.predict(row)[0], 0.0), 1.0))


if __name__ == "__main__":
    _, feats = load_model()
    sample = {f: 5 for f in feats}
    print("Sample prediction:", round(predict_one(sample), 3))
