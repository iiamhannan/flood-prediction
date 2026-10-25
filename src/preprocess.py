"""Preprocessing helpers: split data and build scaled models."""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_data import TARGET, load_data


def get_train_test(sample_size=None, test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test.

    sample_size: use only this many rows (the full dataset has 1.1M rows,
    so sampling keeps training fast on a normal laptop).
    """
    df = load_data().drop(columns=["id"], errors="ignore")
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state)
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_scaled_model(model):
    """Wrap a model with StandardScaler.

    Using a pipeline means the scaler is fitted only on training data
    (no data leakage) and is saved together with the model.
    """
    return make_pipeline(StandardScaler(), model)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test(sample_size=50_000)
    print("Train:", X_train.shape, "Test:", X_test.shape)
