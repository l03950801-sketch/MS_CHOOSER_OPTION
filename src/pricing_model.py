import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Configuration
PROCESSED_DATA_PATH = Path("data/processed_data.csv")
RISK_FREE_ADJUST = 100
VOL_ADJUST = 100
TRADING_DAYS = 252
ROLLING_WINDOW = 20
STRIKE = 110
T_MATURITY = 1/12
T_CHOOSER = 1/24

# WEEK 3: PRICING ERROR COMPARISON ANALYSIS

# BS Model Calculation (Benchmark)
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

# BS Model Validation 
def validate_bs_model() -> None:
    test_price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    print(f"BS Model Validation | Test Call Price: {round(test_price, 4)}")

# Chooser Option Pricing (Formula: CALL+PUT)
def chooser_option_fixed(S: float, K: float, T_choose: float, T_maturity: float, r: float, sigma: float) -> float:
    call = black_scholes(S, K, T_maturity, r, sigma, "call")
    put = black_scholes(S, K, T_maturity, r, sigma, "put")
    return call + put

# Processed Data Conversion
def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").dropna()
    
    df["r"] = df["r"] / RISK_FREE_ADJUST
    df["vol"] = df["vol"] / VOL_ADJUST
    # Actual Annualized Volatility
    df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
    
    return df.dropna()

# Linear Correction Integrating Sentiment Scores
def adjust_volatility(sigma: float, sentiment: float) -> float:
    return sigma * (1 + sentiment * 0.1)

# Evaluation of Linear Correction Model (Sentiment Data)
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return {"MAE": float(round(mae, 4)), "RMSE": float(round(rmse, 4))}

def compute_standardized_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 定义特征 + 目标变量
    features = ["sentiment", "r", "S", "vol"]
    X = df[features].copy()
    y = df["rolling_vol"].copy()

    # 2. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 训练线性回归
    lr = LinearRegression()
    lr.fit(X_scaled, y)

    # 4. 生成结果表
    coef_df = pd.DataFrame({
        "Feature": features,
        "Standardized_Coefficient": np.round(lr.coef_, 4),
        "Abs_Importance": np.round(np.abs(lr.coef_), 4)
    }).sort_values(by="Abs_Importance", ascending=False)

    return coef_df

# Output
def run_comparison_analysis(df: pd.DataFrame) -> None:
    print("WEEK 3: PRICING ERROR COMPARISON ANALYSIS")
    
    benchmark = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], row["rolling_vol"], "call")
        for _, row in df.iterrows()
    ])
    
    # VIX Volatility (Modelled Volatility)
    pred_vix = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], row["vol"], "call")
        for _, row in df.iterrows()
    ])
    
    # Sentiment Adjust 
    pred_sentiment = np.array([
        black_scholes(row["S"], STRIKE, T_MATURITY, row["r"], adjust_volatility(row["vol"], row["sentiment"]), "call")
        for _, row in df.iterrows()
    ])

    metrics_vix = calculate_metrics(benchmark, pred_vix)
    metrics_sentiment = calculate_metrics(benchmark, pred_sentiment)
    
    print(f"BS Model (VIX Volatility)   | MAE: {metrics_vix['MAE']} | RMSE: {metrics_vix['RMSE']}")
    print(f"BS Model (Sentiment Adjust) | MAE: {metrics_sentiment['MAE']} | RMSE: {metrics_sentiment['RMSE']}")
    print("\nComparison Analysis Completed")

# WEEK 4: Limitation Validation
def run_week4_limitation_validation(df: pd.DataFrame, benchmark, pred_vix, pred_sentiment):
    print("WEEK 4: LIMITATION VALIDATION RESULTS")

# 1. High Volatility Failure Mode
    vol_median = df['rolling_vol'].median()
    high_vol_mask = df['rolling_vol'] >= vol_median
    low_vol_mask  = df['rolling_vol'] < vol_median

    print("【1】High vs Low Volatility Regimes (MAE / RMSE)")
    print("High Volatility Periods:")
    print(f"  BSM (VIX):        {calculate_metrics(benchmark[high_vol_mask], pred_vix[high_vol_mask])}")
    print(f"  BSM + Sentiment:  {calculate_metrics(benchmark[high_vol_mask], pred_sentiment[high_vol_mask])}")
    
    print("\nLow Volatility Periods:")
    print(f"  BSM (VIX):        {calculate_metrics(benchmark[low_vol_mask], pred_vix[low_vol_mask])}")
    print(f"  BSM + Sentiment:  {calculate_metrics(benchmark[low_vol_mask], pred_sentiment[low_vol_mask])}")

# 2.sentiment impact gaps
    pos_sent_mask = df['sentiment'] >= 0
    neg_sent_mask = df['sentiment'] < 0

    print("\n--------------------------------------------------")
    print("【2】Positive vs Negative Sentiment (MAE / RMSE)")
    print("Positive Sentiment:")
    print(f"  BSM (VIX):        {calculate_metrics(benchmark[pos_sent_mask], pred_vix[pos_sent_mask])}")
    print(f"  BSM + Sentiment:  {calculate_metrics(benchmark[pos_sent_mask], pred_sentiment[pos_sent_mask])}")
    
    print("\nNegative Sentiment:")
    print(f"  BSM (VIX):        {calculate_metrics(benchmark[neg_sent_mask], pred_vix[neg_sent_mask])}")
    print(f"  BSM + Sentiment:  {calculate_metrics(benchmark[neg_sent_mask], pred_sentiment[neg_sent_mask])}")

    print("\n Limitation validation completed.")

if __name__ == "__main__":
    validate_bs_model()
    
    df = load_processed_data()
    print(f"Processed Data Loaded | Observations: {len(df)}")
    
    sample_price = chooser_option_fixed(df["S"].iloc[0], STRIKE, T_CHOOSER, T_MATURITY, df["r"].iloc[0], df["vol"].iloc[0])
    print(f"Fixed-Date Chooser Option Price: {round(sample_price, 4)}")
    
    run_comparison_analysis(df)

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

    run_week4_limitation_validation(df, benchmark, pred_vix, pred_sentiment)

    print("\n" + "="*60)
    print("LINEAR REGRESSION | STANDARDIZED COEFFICIENTS (FEATURE IMPORTANCE)")
    print("="*60)
    coef_result = compute_standardized_coefficients(df)
    print(coef_result)
    coef_result.to_csv("results/standardized_coefficients.csv", index=False)
    print("\n 标准化回归系数已保存至 results/standardized_coefficients.csv")