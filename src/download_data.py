import yfinance as yf
import pandas as pd
from fredapi import Fred
from pathlib import Path

# Configuration

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

FRED_API_KEY = "926c9e047076f239237763f6a03961da"

OUTPUT_PATH = Path("data/raw_data.csv")

# Download financial data

def download_stock_data():

    print("Downloading JPM stock data...")

    jpm = yf.download("JPM", start=START_DATE, end=END_DATE)

    jpm = jpm[["Close"]]
    jpm.columns = ["jpm_close"]

    return jpm


def download_vix_data():

    print("Downloading VIX data...")

    vix = yf.download("^VIX", start=START_DATE, end=END_DATE)

    vix = vix[["Close"]]
    vix.columns = ["vix"]

    return vix

# Download macroeconomic data

def download_treasury_data():
    print("Downloading Treasury yield data...")
    fred = Fred(api_key=FRED_API_KEY)

    treasury = fred.get_series("DGS10")
    treasury = treasury.to_frame(name="treasury_rate")

    treasury.index = pd.to_datetime(treasury.index)

    return treasury


# Merge datasets

def merge_data(jpm, vix, treasury):

    print("Merging datasets...")

    data = pd.concat([jpm, vix], axis=1)

    data = data.merge(
        treasury,
        left_index=True,
        right_index=True,
        how="left"
    )

    data.reset_index(inplace=True)
    data.rename(columns={"Date": "date"}, inplace=True)

    return data


# Main pipeline

def main():

    print("Starting data pipeline...")

    jpm = download_stock_data()
    vix = download_vix_data()
    treasury = download_treasury_data()

    dataset = merge_data(jpm, vix, treasury)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(OUTPUT_PATH, index=False)

    print("Data saved to:", OUTPUT_PATH)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()