# Data Cleaning & Feature Engineering Specification

## 1. Overview

This document outlines the data cleaning and preprocessing pipeline applied to the raw dataset (week 2).
The objective is to transform raw financial, macroeconomic, and sentiment data into a structured format suitable for quantitative modeling, specifically for option pricing under the Black-Scholes framework.

The cleaned dataset is saved as:

```
data/processed_data.csv
```

---

## 2. Input Data

The raw dataset consists of the following variables:

| Variable      | Description                   |
| ------------- | ----------------------------- |
| date          | Trading date                  |
| jpm_close     | JPM stock closing price       |
| vix           | Market volatility index (VIX) |
| treasury_rate | 10-year US Treasury yield     |
| sentiment     | Daily news sentiment score    |

---

## 3. Data Cleaning Pipeline

The data cleaning process follows a structured pipeline to ensure data quality and consistency.

---

### 3.1 Time-Series Alignment (Trading Calendar Adjustment)

The dataset is aligned to the official NYSE trading calendar using the `pandas_market_calendars` library.

Steps:

* Retrieve valid NYSE trading days
* Reindex dataset to match trading calendar

Purpose:

* Ensure consistency in trading days
* Remove discrepancies caused by weekends and market holidays
* Prepare data for time-series modeling

---

### 3.2 Missing Value Handling

Missing values are addressed using a combination of interpolation and forward/backward filling.

Method:

* Linear interpolation (`method = "linear"`)
* Forward fill (`ffill`)
* Backward fill (`bfill`)

Purpose:

* Preserve time-series continuity
* Avoid loss of observations
* Maintain smooth transitions in financial data

---

### 3.3 Outlier Treatment (Winsorization)

Outliers are handled using winsorization at the 1% level.

Method:

* Values below the 1st percentile are capped
* Values above the 99th percentile are capped

Purpose:

* Reduce the impact of extreme values
* Improve robustness of downstream models
* Stabilize volatility and return calculations

---

## 4. Feature Engineering

After cleaning, the dataset is transformed into model-ready features.

---

### 4.1 Underlying Asset Price (S)

```text
S = jpm_close
```

Description:

* The closing price of JPM stock is renamed to `S`
* Represents the underlying asset price in the option pricing model

---

### 4.2 Log Returns

```text
return = log(S_t) - log(S_{t-1})
```

Description:

* Logarithmic returns are computed from the underlying price

Purpose:

* Preferred in financial modeling due to time-additivity
* Provides a more stable representation of price changes

---

### 4.3 Volatility Proxy

```text
vol = vix
```

Description:

* VIX index is used as a proxy for market-implied volatility

Purpose:

* Serves as the volatility input in the option pricing model
* Reflects forward-looking market uncertainty

---

### 4.4 Risk-Free Rate

```text
r = treasury_rate
```

Description:

* Treasury yield is renamed to `r`

Purpose:

* Represents the risk-free rate required in Black-Scholes pricing

---

### 4.5 Sentiment Indicator

```text
sentiment = daily news sentiment score
```

Description:

* Aggregated sentiment score derived from financial news

Purpose:

* Captures market sentiment as an additional explanatory variable
* Enables analysis of sentiment impact on option pricing

---

## 5. Final Dataset Structure

| Variable  | Description            |
| --------- | ---------------------- |
| date      | Trading date           |
| S         | Underlying asset price |
| return    | Log return             |
| vol       | Volatility proxy (VIX) |
| r         | Risk-free rate         |
| sentiment | News sentiment score   |

---

## 6. Data Quality Checks

After processing, the dataset is validated using:

* Total number of observations
* Missing value count
* Column structure verification
* Date range consistency

Purpose:

* Ensure completeness and correctness of the dataset
* Confirm readiness for modeling

---

## 7. Assumptions

* VIX is an appropriate proxy for implied volatility
* Treasury yield represents the risk-free rate
* Sentiment scores accurately reflect market sentiment
* Linear interpolation adequately approximates missing values

---

## 8. Limitations

1. Winsorization may suppress extreme but meaningful events
2. VIX reflects market-level volatility, not asset-specific volatility
3. Sentiment data may contain noise due to automated classification
4. Daily frequency ignores intra-day market dynamics

---

## 9. Output

The final processed dataset is exported as:

```
data/processed_data.csv
```

This dataset serves as input for the option pricing model implemented in the next stage.
