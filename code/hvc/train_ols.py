import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib


TRAIN_PATH = Path("data/high_value_customer/splits/training_data.csv")
TEST_PATH = Path("data/high_value_customer/splits/test_data.csv")
MODEL_DIR = Path("code/high_value_customer/models")

RANDOM_STATE = 42
THRESHOLD = 0.5   # for binary decision


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["High_Value"])
    y_train = train_df["High_Value"]

    X_test = test_df.drop(columns=["High_Value"])
    y_test = test_df["High_Value"]

    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")

    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)

    y_pred_continuous = ols_model.predict(X_test)

    # Clip predictions to [0, 1] range
    y_pred_prob = np.clip(y_pred_continuous, 0, 1)

    # Convert to binary labels
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {auc:.4f}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ols_model, MODEL_DIR / "currentOlsSolution.pkl")

    print("OLS model saved at:")
    print(MODEL_DIR / "currentOlsSolution.pkl")


if __name__ == "__main__":
    train_and_evaluate()


