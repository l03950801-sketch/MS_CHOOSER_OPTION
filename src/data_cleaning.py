import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal

# Path Configuration
RAW_DATA_PATH = Path("data/raw_data.csv")
CLEANED_DATA_PATH = Path("data/processed_data.csv")

# Cleaning Parameters
INTERPOLATION_METHOD = "linear"
IQR_MULTIPLIER = 1.5
WINSORIZE_LIMIT = 0.01
EXCHANGE = "NYSE"

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    print(f"Raw data loaded | Total trading days: {len(df)}")
    return df

def align_time_series(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df.index.min()
    end_date = df.index.max()
    nyse_cal = mcal.get_calendar(EXCHANGE)
    trading_days = nyse_cal.valid_days(start_date, end_date).tz_localize(None)

    df_aligned = df.reindex(trading_days)
    df_aligned.index.name = "date"
    return df_aligned

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.interpolate(method=INTERPOLATION_METHOD, limit_direction="both")
    df_clean = df_clean.ffill().bfill()
    return df_clean

def detect_outliers_iqr(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df_clean[col] = df_clean[col].clip(
            lower=df_clean[col].quantile(WINSORIZE_LIMIT),
            upper=df_clean[col].quantile(1 - WINSORIZE_LIMIT)
        )
    return df_clean

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df_transformed = df.copy()
    df_transformed.rename(columns={"jpm_close": "S"}, inplace=True)
    df_transformed["return"] = np.log(df_transformed["S"]).diff()
    df_transformed.rename(columns={"vix": "vol", "treasury_rate": "r"}, inplace=True)
    df_transformed = df_transformed[["S", "return", "vol", "r", "sentiment"]]
    return df_transformed

# new added on 14th Apr, 2026
def print_descriptive_stats(df: pd.DataFrame, title: str):
    print(f"\n=== {title} ===")
    print(df.describe().round(4))

def data_quality_check(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print(" DATA QUALITY CHECK & STATISTICS COMPARISON")
    print("="*60)
    
    print(f"Total rows: {len(df_after)}")
    print(f"Missing values before cleaning: {df_before.isnull().sum().sum()}")
    print(f"Missing values after cleaning: {df_after.isnull().sum().sum()}")
    
    print_descriptive_stats(df_before, "Descriptive Statistics BEFORE Cleaning")
    
    print_descriptive_stats(df_after, "Descriptive Statistics AFTER Cleaning")
    
    print("\n✅ Data quality check completed.")

# -------------------------------------------------------------------------

def clean_data_pipeline() -> pd.DataFrame:
    print("="*60)
    print("Week2 Data Cleaning Pipeline Started")
    print("="*60)
    
    df_raw = load_raw_data()
    df_aligned = align_time_series(df_raw)
    df_no_missing = handle_missing_values(df_aligned)
    df_clean = handle_outliers(df_no_missing)
    df_final = transform_features(df_clean)
    
    data_quality_check(df_raw, df_final)
    
    return df_final

def save_cleaned_data(df: pd.DataFrame) -> None:
    df.to_csv(CLEANED_DATA_PATH)
    print(f"\nCleaned data saved to: {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    cleaned_dataset = clean_data_pipeline()
    save_cleaned_data(cleaned_dataset)
    
    print("\n" + "="*60)
    print("Week2 Data Cleaning Pipeline Completed Successfully")
    print("="*60)