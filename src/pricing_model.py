import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# Configuration
PROCESSED_DATA_PATH = Path("data/processed_data.csv")
RISK_FREE_ADJUST = 100
VOL_ADJUST = 100
TRADING_DAYS = 252
ROLLING_WINDOW = 20
STRIKE = 110
T_MATURITY = 1/12
T_CHOOSER = 1/24

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    if sigma <= 0 or T <= 0:
        return np.nan
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("Option type must be 'call' or 'put'")

def validate_bs_model() -> None:
    test_price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    print(f"BS Model Validation | Test Call Price: {round(test_price, 4)}")

def chooser_option_fixed(S: float, K: float, T_choose: float, T_maturity: float, r: float, sigma: float) -> float:
    call = black_scholes(S, K, T_maturity, r, sigma, "call")
    put = black_scholes(S, K, T_maturity, r, sigma, "put")
    return call + put

def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").dropna()
    
    df["r"] = df["r"] / RISK_FREE_ADJUST
    df["vol"] = df["vol"] / VOL_ADJUST
    df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
    
    return df.dropna()

def adjust_volatility(sigma: float, sentiment: float) -> float:
    return sigma * (1 + sentiment * 0.1)

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}

def run_comparison_analysis(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("WEEK 3: PRICING ERROR COMPARISON ANALYSIS")
    print("="*60)
    
    benchmark = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], row["rolling_vol"], "call")
        for _, row in df.iterrows()
    ])
    
    pred_vix = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], row["vol"], "call")
        for _, row in df.iterrows()
    ])
    
    pred_sentiment = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], adjust_volatility(row["vol"], row["sentiment"]), "call")
        for _, row in df.iterrows()
    ])

    metrics_vix = calculate_metrics(benchmark, pred_vix)
    metrics_sentiment = calculate_metrics(benchmark, pred_sentiment)
    
    print(f"BS Model (VIX Volatility)   | MAE: {metrics_vix['MAE']} | RMSE: {metrics_vix['RMSE']}")
    print(f"BS Model (Sentiment Adjust) | MAE: {metrics_sentiment['MAE']} | RMSE: {metrics_sentiment['RMSE']}")
    print("\nComparison Analysis Completed")

if __name__ == "__main__":
    validate_bs_model()
    
    df = load_processed_data()
    print(f"Processed Data Loaded | Observations: {len(df)}")
    
    sample_price = chooser_option_fixed(df["S"].iloc[0], STRIKE, T_CHOOSER, T_MATURITY, df["r"].iloc[0], df["vol"].iloc[0])
    print(f"Fixed-Date Chooser Option Price: {round(sample_price, 4)}")
    
    run_comparison_analysis(df)