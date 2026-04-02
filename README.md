# Chooser Option Pricing with Market & Sentiment Data

## 1. Project Overview

This project builds a financial data pipeline and applies it to the analysis of a Chooser Option Pricing, a type of exotic derivative that allows the holder to choose whether the option will be a call or a put at a future date.

The objective is to integrate:

* Financial data
* Macroeconomic indicators
* News sentiment data

to support quantitative finance research and option pricing.

---

## 2. Key Features

* Automated data pipeline using APIs
* Integration of financial, macroeconomic, and sentiment data
* Clean and structured dataset for quantitative modeling
* Designed for extension into option pricing models (e.g., Black-Scholes, Monte Carlo)

---

## 3. Data Sources

| Category      | Source        | Description              |
| ------------- | ------------- | ------------------------ |
| Stock Price   | Yahoo Finance | JPM closing prices       |
| Volatility    | VIX Index     | Market volatility proxy  |
| Interest Rate | FRED          | 10Y Treasury yield       |
| Sentiment     | Alpha Vantage | Financial news sentiment |

---

## 4. Project Structure

```
MS_CHOOSER_OPTION
│
├── src
│   └── download_data.py        # Data pipeline
│
├── data
│   └── raw_data.csv           # Final dataset
│
├── docs
│   └── data_specification.md  # Data documentation
│
└── README.md
```


## 5. Output

The pipeline generates:

```
data/raw_data.csv
```

Dataset structure:

| date | jpm_close | vix | treasury_rate | sentiment |
| ---- | --------- | --- | ------------- | --------- |

---

## 6. Future Work

* Implement Chooser Option pricing model
* Apply Black-Scholes framework
* Conduct data cleaning and data wrangling to enhance data integrity
* Explore sentiment-driven volatility prediction

---
