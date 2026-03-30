import yfinance as yf
import pandas as pd
from fredapi import Fred
from pathlib import Path
import requests

# Configuration

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

FRED_API_KEY = "926c9e047076f239237763f6a03961da"
AV_API_KEY = "RXP2APE979NRUVVT"

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


# Download sentiment data

def download_sentiment_data():

    print("Downloading news sentiment data...")

    url = (
        "https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&tickers=JPM"
        f"&time_from=20180101T0000"
        f"&time_to=20241231T0000"
        f"&apikey={AV_API_KEY}"
    )

    response = requests.get(url).json()

    articles = response.get("feed", [])

    sentiment_data = []

    for article in articles:

        date = article["time_published"][:8]

        score = article["overall_sentiment_score"]

        sentiment_data.append({
            "date": pd.to_datetime(date),
            "sentiment": score
        })

    sentiment_df = pd.DataFrame(sentiment_data)

    sentiment_df = sentiment_df.groupby("date").mean()

    return sentiment_df


# Merge datasets

def merge_data(jpm, vix, treasury, sentiment):

    print("Merging datasets...")

    data = pd.concat([jpm, vix], axis=1)

    data = data.merge(
        treasury,
        left_index=True,
        right_index=True,
        how="left"
    )

    data = data.merge(
        sentiment,
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

    sentiment = download_sentiment_data()

    dataset = merge_data(jpm, vix, treasury, sentiment)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(OUTPUT_PATH, index=False)

    print("Data saved to:", OUTPUT_PATH)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()