import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/high_value_customer/raw/ecommerce_scraped.csv")
PROCESSED_DIR = Path("data/high_value_customer/processed")
SPLIT_DIR = Path("data/high_value_customer/splits")

RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_VALUE_QUANTILE = 0.75


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["Lifetime_Value"].quantile(HIGH_VALUE_QUANTILE)
    df["High_Value"] = (df["Lifetime_Value"] >= threshold).astype(int)
    df = df.drop(columns=["Lifetime_Value"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numerical_cols:
        if col != "High_Value":
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numerical_cols = [c for c in numerical_cols if c != "High_Value"]

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def split_and_save(df: pd.DataFrame):
 
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


    df.to_csv(PROCESSED_DIR / "joint_data_collection.csv", index=False)

    X = df.drop(columns=["High_Value"])
    y = df["High_Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


    X_train.assign(High_Value=y_train).to_csv(
        SPLIT_DIR / "training_data.csv", index=False
    )

    X_test.assign(High_Value=y_test).to_csv(
        SPLIT_DIR / "test_data.csv", index=False
    )


    activation_sample = X_test.sample(1, random_state=RANDOM_STATE)
    activation_sample.to_csv(
        SPLIT_DIR / "activation_data.csv", index=False
    )


# Pipeline Runner


def run_preprocessing_pipeline():
    print("Loading raw data...")
    df = load_data()

    print("Creating target variable...")
    df = create_target(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Scaling numerical features...")
    df = scale_numerical_features(df)

    print("Encoding categorical features...")
    df = encode_categorical_features(df)

    print("Splitting data and saving CSV files...")
    split_and_save(df)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
