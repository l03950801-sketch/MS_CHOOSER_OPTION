# Week5 ML 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ======================
# 全局固定参数
# ======================
TRADING_DAYS = 252
ROLLING_WINDOW = 20
STRIKE = 110
T_MATURITY = 1/12
EPS = 1e-8
DIR_THRESHOLD = 0.001

# ======================
# 数据预处理
# ======================
df = pd.read_csv("data/processed_data.csv")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# 个股20日滚动波动率
df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
# 预测目标：下一日波动率（t+1）
df["target_vol_t1"] = df["rolling_vol"].shift(-1)

# BSM定价函数
def black_scholes(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# 合成远期期权价格（t+1）
df['synthetic_price_t'] = df.apply(lambda x: black_scholes(x['S'], STRIKE, T_MATURITY, x['r'], x['rolling_vol']), axis=1)
df['synthetic_price_t1'] = df['synthetic_price_t'].shift(-1)
df = df.dropna().reset_index(drop=True)

# ======================
# 特征工程
# ======================
def build_features(data):
    df = data.copy()
    df['sent_lag1'] = df['sentiment'].shift(1)    # 滞后情绪
    df['S_lag1'] = df['S'].shift(1)               # 滞后股价
    df['r_lag1'] = df['r'].shift(1)               # 滞后利率
    df['rv_lag1'] = df['rolling_vol'].shift(1)    # 个股滞后波动率
    df['sent_ma5'] = df['sentiment'].rolling(5).mean() # 情绪趋势
    return df.dropna()

df_final = build_features(df)
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']

# ======================
# 时序拆分
# ======================
n = len(df_final)
train_size = int(0.7 * n)
df_train = df_final.iloc[:train_size].copy()
df_test = df_final.iloc[train_size:].copy()

X_train, X_test = df_train[FEATURES], df_test[FEATURES]
y_train, y_test = df_train['target_vol_t1'], df_test['target_vol_t1']
benchmark_test = df_test['synthetic_price_t1']

# ======================
# 评估函数
# ======================
def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + EPS))) * 100
    
    if len(y_true) < 2:
        dir_acc = 0.0
    else:
        y_true_diff = y_true[1:] - y_true[:-1]
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        
        true_dir = np.sign(np.clip(y_true_diff, -DIR_THRESHOLD, DIR_THRESHOLD))
        pred_dir = np.sign(np.clip(y_pred_diff, -DIR_THRESHOLD, DIR_THRESHOLD))
        dir_acc = np.mean(true_dir == pred_dir)
        
    return mse, mae, mape, dir_acc

# ======================
# Naive Baseline
# ======================
print("="*80)
print("统一时序：预测 t+1 个股波动率 → 定价 t+1 期权")
print("="*80)

print("\n【Naive Baseline】σ_{t+1} = σ_t")
y_naive_pred = df_test['rv_lag1']
naive_mse, naive_mae, naive_mape, naive_dir = evaluate(y_test, y_naive_pred)
print(f"Naive | MSE: {naive_mse:.6f} | MAE: {naive_mae:.6f} | MAPE: {naive_mape:.2f}% | 方向准确率: {naive_dir:.2%}")

# ======================
# 机器学习波动率预测
# ======================
print("\n【Approach 1】")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

rf_mse, rf_mae, rf_mape, rf_dir = evaluate(y_test, rf_pred)
xgb_mse, xgb_mae, xgb_mape, xgb_dir = evaluate(y_test, xgb_pred)

print(f"随机森林 | MSE: {rf_mse:.6f} | MAE: {rf_mae:.6f} | MAPE: {rf_mape:.2f}% | 方向准确率: {rf_dir:.2%}")
print(f"XGBoost   | MSE: {xgb_mse:.6f} | MAE: {xgb_mae:.6f} | MAPE: {xgb_mape:.2f}% | 方向准确率: {xgb_dir:.2%}")

# 最优模型
best_model = rf if rf_mse < xgb_mse else xgb
best_vol_pred = rf_pred if rf_mse < xgb_mse else xgb_pred
improvement = (naive_mse - min(rf_mse, xgb_mse)) / naive_mse * 100
print(f"\n 最优模型: {'随机森林' if rf_mse < xgb_mse else 'XGBoost'} | 相对Baseline提升: {improvement:.2f}%")

# 特征重要性
print("\n 波动率预测特征重要性")
imp_df = pd.DataFrame({'feature': FEATURES, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
print(imp_df)

# ======================
# 远期期权定价
# ======================
pred_prices = [black_scholes(df_test.iloc[i]['S'], STRIKE, T_MATURITY, df_test.iloc[i]['r'], best_vol_pred[i]) for i in range(len(df_test))]
price_mse = mean_squared_error(benchmark_test, pred_prices)
print(f"\n【Approach1】MSE: {price_mse:.6f}")

# ======================
# E2E 定价模型
# ======================
print("\n" + "="*60)
print("【Approach2 E2E】")
print("="*60)

X_e2e_train, X_e2e_test = df_train[FEATURES], df_test[FEATURES]
y_e2e_train, y_e2e_test = df_train['synthetic_price_t1'], df_test['synthetic_price_t1']

# 模型训练与评估
lr_e2e = LinearRegression()
lr_e2e.fit(X_e2e_train, y_e2e_train)
lr_pred = lr_e2e.predict(X_e2e_test)

xgb_e2e = XGBRegressor(random_state=42)
xgb_e2e.fit(X_e2e_train, y_e2e_train)
xgb_e2e_pred = xgb_e2e.predict(X_e2e_test)

lr_mse = mean_squared_error(y_e2e_test, lr_pred)
xgb_mse = mean_squared_error(y_e2e_test, xgb_e2e_pred)

best_e2e_name = "XGBoost" if xgb_mse < lr_mse else "线性回归"
e2e_mse = min(lr_mse, xgb_mse)

print(f"E2E最优模型: {best_e2e_name} | 定价MSE: {e2e_mse:.6f}")

print("\n" + "="*80)
print(" 全部完成")
print("="*80)