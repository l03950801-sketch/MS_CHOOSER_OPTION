import yfinance as yf
import pandas as pd
from fredapi import Fred
from pathlib import Path
import requests
from datetime import datetime, timedelta
import time

# configuration
START_DATE = "2018-01-01"
END_DATE = "2019-12-31"

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


def download_sentiment_data():
    print("Downloading JPM news sentiment data (2018-2019, monthly batch)...")
    all_articles = []
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")


    current = start
    while current <= end:

        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)
        period_end = next_month - timedelta(days=1)


        time_from = current.strftime("%Y%m%dT%H%M")
        time_to = period_end.strftime("%Y%m%dT%H%M")


        url = (
            "https://www.alphavantage.co/query?"
            "function=NEWS_SENTIMENT"
            "&tickers=JPM"
            f"&time_from={time_from}"
            f"&time_to={time_to}"
            f"&apikey={AV_API_KEY}"
            "&limit=1000"  
        )

        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            articles = data.get("feed", [])
            all_articles.extend(articles)
            print(f"✅ {current.strftime('%Y-%m')} | numbers of news obtained: {len(articles)}")
        except Exception as e:
            print(f"NO! {current.strftime('%Y-%m')} request fails: {str(e)}")


        time.sleep(12)
        current = next_month


    sentiment_data = []
    for article in all_articles:
        try:
      
            date_str = article["time_published"][:8]
            date = pd.to_datetime(date_str, format="%Y%m%d")
            score = float(article["overall_sentiment_score"])
            sentiment_data.append({"date": date, "sentiment": score})
        except:
            continue

    if not sentiment_data:
        return pd.DataFrame(columns=["date", "sentiment"]).set_index("date")

    sentiment_df = pd.DataFrame(sentiment_data)

    sentiment_df = sentiment_df.groupby("date")["sentiment"].mean().to_frame()
    return sentiment_df


# Merge datasets
def merge_data(jpm, vix, treasury, sentiment):
    print("Merging datasets...")

    data = pd.concat([jpm, vix], axis=1)

    data = data.merge(treasury, left_index=True, right_index=True, how="left")

    data = data.merge(sentiment, left_index=True, right_index=True, how="left")

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

    print(f"\ndata has been saved to: {OUTPUT_PATH}")
    print(f"total numbers of rows of data: {len(dataset)}")
    print("✅ Pipeline success")


if __name__ == "__main__":
    main()