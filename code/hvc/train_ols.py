import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Paths
# =========================

TRAIN_PATH = Path("data/high_value_customer/splits/training_data.csv")
TEST_PATH = Path("data/high_value_customer/splits/test_data.csv")
MODEL_DIR = Path("code/high_value_customer/models")

DOC_DIR = Path("documentation")
DOC_DIR.mkdir(exist_ok=True)

# =========================
# Config
# =========================

RANDOM_STATE = 42
THRESHOLD = 0.5


# =========================
# Data Loading
# =========================

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["High_Value"])
    y_train = train_df["High_Value"]

    X_test = test_df.drop(columns=["High_Value"])
    y_test = test_df["High_Value"]

    return X_train, X_test, y_train, y_test


# =========================
# Training + Evaluation
# =========================

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")

    # -------------------------
    # Train OLS
    # -------------------------
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)

    # -------------------------
    # Predictions
    # -------------------------
    y_pred_continuous = ols_model.predict(X_test)
    y_pred_prob = np.clip(y_pred_continuous, 0, 1)
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {auc:.4f}")

    # -------------------------
    # Save model
    # -------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ols_model, MODEL_DIR / "currentOlsSolution.pkl")

    print("OLS model saved at:")
    print(MODEL_DIR / "currentOlsSolution.pkl")

    # =========================
    # Visualizations
    # =========================

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"OLS (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – OLS")
    plt.legend()
    plt.savefig(DOC_DIR / "ols_roc_curve.png")
    plt.close()

    # 2. Probability Distribution
    plt.figure()
    sns.histplot(y_pred_prob[y_test == 0], label="Class 0", kde=True)
    sns.histplot(y_pred_prob[y_test == 1], label="Class 1", kde=True)
    plt.xlabel("Predicted Probability")
    plt.title("OLS Prediction Probability Distribution")
    plt.legend()
    plt.savefig(DOC_DIR / "ols_probability_distribution.png")
    plt.close()

    # 3. Residual Analysis
    residuals = y_test - y_pred_prob

    plt.figure()
    plt.scatter(y_pred_prob, residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("OLS Residuals vs Predicted Values")
    plt.savefig(DOC_DIR / "ols_residuals.png")
    plt.close()

    print("OLS visualizations saved to /documentation")


# =========================
# Main
# =========================

if __name__ == "__main__":
    train_and_evaluate()