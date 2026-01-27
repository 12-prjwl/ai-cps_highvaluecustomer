import requests
import pandas as pd
from pathlib import Path
import io



RAW_DATA_URL = (
    "https://raw.githubusercontent.com/12-prjwl/AI-CPS_HighValueCustomer/refs/heads/main/data/high_value_customer/raw/ecommerce_customer_churn_dataset.csv"
)

OUTPUT_DIR = Path("data/high_value_customer/raw")
OUTPUT_FILE = OUTPUT_DIR / "ecommerce_scraped.csv"


def scrape_dataset():

    response = requests.get(RAW_DATA_URL)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    print(f"Dataset downloaded with shape: {df.shape}")
    return df


def save_raw_dataset(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Raw scraped dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    dataset = scrape_dataset()
    save_raw_dataset(dataset)
