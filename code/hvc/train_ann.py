import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, roc_auc_score


TRAIN_PATH = Path("data/high_value_customer/splits/training_data.csv")
TEST_PATH = Path("data/high_value_customer/splits/test_data.csv")
MODEL_DIR = Path("code/high_value_customer/models")

EPOCHS = 30         
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["High_Value"])
    y_train = train_df["High_Value"]

    X_test = test_df.drop(columns=["High_Value"])
    y_test = test_df["High_Value"]

    return X_train, X_test, y_train, y_test



def build_ann(input_dim: int):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def train_and_evaluate():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")

    model = build_ann(X_train.shape[1])

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Evaluating model...")
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {auc:.4f}")

    # Saving the model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "currentAiSolution.h5")

    print("Model saved at:")
    print(MODEL_DIR / "currentAiSolution.h5")



if __name__ == "__main__":
    train_and_evaluate()
