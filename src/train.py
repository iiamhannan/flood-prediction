"""Train Linear Regression, Random Forest and XGBoost, compare them,
plot feature importance and save the best model."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import get_train_test, make_scaled_model

SAMPLE_SIZE = 200_000  # set to None to use all rows (slower)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "R2": r2_score(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
    }


def plot_feature_importance(model, feature_names, path):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(8, 7))
    imp.plot(kind="barh", color="steelblue")
    plt.title("Feature importance (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = get_train_test(sample_size=SAMPLE_SIZE)

    models = {
        "Linear Regression": make_scaled_model(LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            n_jobs=-1, random_state=42,
        ),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({"Model": name, **evaluate(model, X_test, y_test)})

    results = pd.DataFrame(rows).round(4).sort_values("R2", ascending=False)
    print("\n", results.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    plot_feature_importance(
        fitted["XGBoost"], X_train.columns, RESULTS_DIR / "feature_importance.png"
    )

    best_name = results.iloc[0]["Model"]
    print(f"\nBest model: {best_name}")
    joblib.dump(
        {"model": fitted[best_name], "features": list(X_train.columns), "name": best_name},
        MODELS_DIR / "best_model.joblib",
    )


if __name__ == "__main__":
    main()
