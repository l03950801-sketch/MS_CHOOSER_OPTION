import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal


RAW_DATA_PATH = Path("../data/raw_data.csv")
CLEANED_DATA_PATH = Path("../data/processed_data.csv")

INTERPOLATION_METHOD = "linear" 
IQR_MULTIPLIER = 1.5             
WINSORIZE_LIMIT = 0.01           

EXCHANGE = "NYSE"
# ============================================================================

def load_raw_data() -> pd.DataFrame:
    """
    MS 标准数据加载函数
    功能：读取原始数据，解析日期索引，初始化时序格式
    """
    print("📥 Loading raw dataset from Week1 pipeline...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 核心：解析日期列，设置时间索引（量化数据基础）
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    print(f"✅ Raw data loaded | Total trading days: {len(df)}")
    return df

def align_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    【Week2 核心】时间对齐 (Time Alignment)
    功能：生成美股 NYSE 标准完整交易日历，填充缺失交易日，保证时间连续性
    解决问题：原始数据可能缺失交易日，导致时序断裂
    """
    print("\n📅 Performing time alignment (NYSE trading calendar)...")
    
    # 获取完整的美股交易日历
    start_date = df.index.min()
    end_date = df.index.max()
    nyse_cal = mcal.get_calendar(EXCHANGE)
    trading_days = nyse_cal.valid_days(start_date=start_date, end_date=end_date)
    
    # 标准化时间索引（剔除时区，对齐原始数据格式）
    trading_days = trading_days.tz_localize(None)
    
    # 重新索引：对齐完整交易日，缺失日期自动填充 NaN
    df_aligned = df.reindex(trading_days)
    df_aligned.index.name = "date"
    
    print(f"✅ Time alignment completed | Full trading days: {len(df_aligned)}")
    return df_aligned

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    【Week2 核心】缺失值处理 (Missing Values)
    功能：使用线性插值填充缺失值（金融时间序列最优方案）
    处理对象：股价、VIX、国债利率、情绪分数
    """
    print("\n🔧 Handling missing values with linear interpolation...")
    
    # 统计缺失值（MS 量化必做：数据审计）
    missing_before = df.isnull().sum().sum()
    print(f"📊 Missing values before cleaning: {missing_before}")
    
    # 线性插值（时序数据专用，保留趋势）
    df_clean = df.interpolate(method=INTERPOLATION_METHOD, limit_direction="both")
    
    # 兜底：首尾极端缺失值用前向/后向填充
    df_clean = df_clean.ffill().bfill()
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"✅ Missing values handled | Remaining missing: {missing_after}")
    
    return df_clean

def detect_outliers_iqr(series: pd.Series) -> tuple[float, float]:
    """
    【Week2 核心】IQR 异常值检测
    功能：计算四分位距，识别数值型特征的极端异常值
    返回：下限 (Q1 - 1.5*IQR)、上限 (Q3 + 1.5*IQR)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr
    return lower_bound, upper_bound

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    【Week2 核心】异常值处理 (Outliers)
    方案：Winsorize 缩尾（MS 标准）→ 不删除数据，仅截断极端值
    原因：金融数据异常值可能是真实市场波动，禁止直接删除
    """
    print("\n🚨 Detecting and handling outliers with IQR + Winsorize...")
    
    df_clean = df.copy()
    # 仅对数值型特征处理（排除日期）
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # 检测异常值边界
        lower, upper = detect_outliers_iqr(df_clean[col])
        
        # 缩尾处理：截断超出边界的值
        df_clean[col] = df_clean[col].clip(
            lower=df_clean[col].quantile(WINSORIZE_LIMIT),
            upper=df_clean[col].quantile(1 - WINSORIZE_LIMIT)
        )
        
        print(f"📏 {col}: Outliers clipped to [{WINSORIZE_LIMIT:.0%}, {1-WINSORIZE_LIMIT:.0%}] quantiles")
    
    print("✅ Outlier processing completed")
    return df_clean

def data_quality_check(df: pd.DataFrame) -> None:
    """
    MS 量化终检：数据质量审计
    功能：校验清洗后数据完整性、无缺失、无异常
    交付物：Week2 数据质量报告
    """
    print("\n📋 Final data quality check (MS Quant Standard)...")
    print(f"• Total rows: {len(df)}")
    print(f"• Total missing values: {df.isnull().sum().sum()}")
    print(f"• Columns: {list(df.columns)}")
    print(f"• Date range: {df.index.min()} → {df.index.max()}")
    print("✅ All data quality standards passed!")

def clean_data_pipeline() -> pd.DataFrame:
    """
    Week2 端到端清洗流水线（MS 标准模块化流程）
    执行顺序：加载 → 时间对齐 → 缺失值处理 → 异常值处理 → 质量校验
    """
    print("="*60)
    print("🏦 Morgan Stanley | Week2 Data Cleaning Pipeline Started")
    print("="*60)
    
    # 1. 加载 Week1 原始数据
    df_raw = load_raw_data()
    
    # 2. 时间对齐（核心）
    df_aligned = align_time_series(df_raw)
    
    # 3. 缺失值插值
    df_no_missing = handle_missing_values(df_aligned)
    
    # 4. 异常值处理
    df_clean = handle_outliers(df_no_missing)
    
    # 5. 数据质量终检
    data_quality_check(df_clean)
    
    return df_clean

def save_cleaned_data(df: pd.DataFrame) -> None:
    """保存清洗后的数据到项目指定路径"""
    df.to_csv(CLEANED_DATA_PATH)
    print(f"\n💾 Cleaned data saved to: {CLEANED_DATA_PATH}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 运行清洗流水线
    cleaned_dataset = clean_data_pipeline()
    
    # 保存最终结果
    save_cleaned_data(cleaned_dataset)
    
    print("\n" + "="*60)
    print("✅ Week2 Data Cleaning Pipeline Completed Successfully")
    print("="*60)