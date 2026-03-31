1. Overview
This project collects a dataset on chooser option pricing for Morgan Stanley. It combines financial market data (JPM), macroeconomic variables, and news sentiment indicators.

The data covers the period 2018-01-01 to 2024-12-31 and is collected through multiple APIs.

Final dataset output:
data/raw_data.csv

2. Data Sources 
2.1 Stock Price Data
Source: Yahoo Finance (via yfinance API)
Asset: JPMorgan Chase & Co.

Variables:
jpm_close, which coollects daily closing price of JPM stock with daily frequency. 

Purpose:
Used as the underlying asset price in option pricing models.

2.2 Market Volatility
Source: Yahoo Finance (via yfinance API)

Index: Volatility Index (VIX)

Variables: VIX, which represents daily closing value of the VIX index

Purpose:
Used as a proxy for market-implied volatility.

2.3 Risk-Free Interest Rate
Source: Federal Reserve Economic Data (via FRED API)

Indicator:
10-Year Treasury Constant Maturity Rate (DGS10)

Variables:
treasury_rate, which tracks daily 10-year US Treasury yield

Purpose:
Used as the risk-free interest rate input in option pricing models.

2.4 Sentiment Data

Source: Alpha Vantage News Sentiment API (via AV API)

Function:
NEWS_SENTIMENT

Ticker:
JPM

Variables:
sentiment, which tracks average daily news sentiment score under "feed"

Sentiment Score Range:
-1 = Bearish sentiment
0 = Neutral sentiment
+1 = Bullish sentiment

Construction Method:
Financial news articles mentioning JPM are retrieved using the API.
Each article includes an overall_sentiment_score.
Sentiment scores are aggregated by date using the daily average.

3. Data Processing Pipeline

The dataset is constructed using the following steps:

Download stock price data
Download volatility index data
Download Treasury yield data
Download financial news sentiment
Merge datasets using date as the key
Export the final dataset to CSV format and save it to the data/raw_data.csv

5. Data Limitations

Several limitations should be considered:

Failure to capture all the sentiment data from 2018 to 2024 due to API request limitation, leading to weak data integrity. 
News sentiment scores may contain noise due to automated sentiment classification.
Some days may contain missing sentiment data due to limited news coverage.
Treasury yield data may include non-trading days and require alignment with market data.
Data cleaning and wrangling have not been finished yet.

6. Intended Usage

The dataset is designed for research applications including:
option pricing analysis
volatility modeling
sentiment-based financial prediction
quantitative finance research