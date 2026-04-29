# Week5 ML
# Approach1: 波动率预测→BSM定价
# Approach2: 纯正E2E定价

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 基础配置
TRADING_DAYS = 252
ROLLING_WINDOW = 20
STRIKE = 110
T_MATURITY = 1/12

# 加载数据
df = pd.read_csv("data/processed_data.csv")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# 滚动波动率
df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
df = df.dropna().reset_index(drop=True)

# 生成期权价格标签(仅用于训练)
def black_scholes_label(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
df['K'] = df['S']
df['T'] = 0.5
df['option_price'] = df.apply(lambda x: black_scholes_label(x['S'],x['K'],x['T'],x['r'],x['vol']), axis=1)

# BSM定价函数
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# 基准价格
df['benchmark_price'] = df.apply(lambda row: black_scholes(row['S'], STRIKE, T_MATURITY, row['r'], row['rolling_vol']), axis=1)

# ======================
# 2. 特征构建
# ======================
def build_features_lag1(data):
    df = data.copy()
    df['sent_lag1'] = df['sentiment'].shift(1)
    df['vol_lag1'] = df['vol'].shift(1)
    df['vol_ma5'] = df['vol'].rolling(5).mean()
    df['sent_ma5'] = df['sentiment'].rolling(5).mean()
    df['r_feature'] = df['r']
    df['S_feature'] = df['S']
    return df.dropna()

def build_features_lag2(data):
    df = data.copy()
    df['sent_lag2'] = df['sentiment'].shift(2)
    df['vol_lag2'] = df['vol'].shift(2)
    df['vol_ma5'] = df['vol'].rolling(5).mean()
    df['sent_ma5'] = df['sentiment'].rolling(5).mean()
    df['r_feature'] = df['r']
    df['S_feature'] = df['S']
    return df.dropna()

# 最优滞后选择
def evaluate_lag(df, lag):
    data = build_features_lag1(df) if lag == 1 else build_features_lag2(df)
    feats = ['vol_ma5','sent_ma5','r_feature','S_feature','sent_lag1','vol_lag1'] if lag ==1 else ['vol_ma5','sent_ma5','r_feature','S_feature','sent_lag2','vol_lag2']
    X, y = data[feats], data['vol']
    split = int(0.7*len(X))
    model = RandomForestRegressor(random_state=42)
    model.fit(X[:split], y[:split])
    return mean_squared_error(y[split:], model.predict(X[split:]))

mse1 = evaluate_lag(df,1)
mse2 = evaluate_lag(df,2)
best_lag = 1 if mse1 < mse2 else 2

print("="*60)
print(f"滞后1天 MSE: {mse1:.6f}")
print(f"滞后2天 MSE: {mse2:.6f}")
print(f"最优选择：滞后 {best_lag} 天")
print("="*60)

# 最终特征
if best_lag ==1:
    df_final = build_features_lag1(df)
else:
    df_final = build_features_lag2(df)

# 时序分割
n = len(df_final)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
df_train = df_final.iloc[:train_size].copy()
df_test  = df_final.iloc[train_size+val_size:].copy()

print(f"训练集:{len(df_train)} | 测试集:{len(df_test)}")
print("="*60)

# 自适应特征
def get_adaptive_params(train_data):
    vol_median = train_data['rolling_vol'].median()
    corr_high = train_data[train_data['rolling_vol'] >= vol_median][['sentiment', 'vol']].corr().iloc[0,1]
    corr_low = train_data[train_data['rolling_vol'] < vol_median][['sentiment', 'vol']].corr().iloc[0,1]
    return vol_median, np.clip(corr_high,0,0.2), np.clip(corr_low,0,0.05)

vol_median, w_high, w_low = get_adaptive_params(df_train)

def add_adaptive_features(df):
    df['adaptive_vol'] = df.apply(lambda row: row['vol']*(1+row['sentiment']*w_high) if row['rolling_vol']>=vol_median else row['vol']*(1+row['sentiment']*w_low), axis=1)
    df['high_vol_flag'] = (df['rolling_vol'] >= vol_median).astype(int)
    return df

df_train = add_adaptive_features(df_train)
df_test = add_adaptive_features(df_test)

# 特征集合
features = [f'sent_lag{best_lag}',f'vol_lag{best_lag}','vol_ma5','sent_ma5','r_feature','S_feature','adaptive_vol','high_vol_flag']
X_train, X_test = df_train[features], df_test[features]
y_train, y_test = df_train['vol'], df_test['vol']
benchmark_test = df_test['benchmark_price']

# ======================
# Approach1 波动率预测+BSM
# ======================
print("\n【Approach 1】波动率预测")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
mse_rf = mean_squared_error(y_test, rf.predict(X_test))
print(f"随机森林 MSE: {mse_rf:.6f}")

xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
mse_xgb = mean_squared_error(y_test, xgb.predict(X_test))
print(f"XGBoost MSE: {mse_xgb:.6f}")

best_vol_pred = rf.predict(X_test) if mse_rf < mse_xgb else xgb.predict(X_test)
print(f"✅ 最优模型：{'随机森林' if mse_rf < mse_xgb else 'XGBoost'}")

# BSM定价
pred_prices = [black_scholes(df_test.iloc[i]['S'], STRIKE, T_MATURITY, df_test.iloc[i]['r'], best_vol_pred[i]) for i in range(len(df_test))]
print(f"\nApproach1 定价 MSE: {mean_squared_error(benchmark_test, pred_prices):.6f}")

# ======================
# Approach2 E2E
# ======================
print("\n" + "="*60)
print("Approach2 E2E定价")
print("="*60)

# 滞后情绪 + 情绪均线
E2E_FEATURES = [f'sent_lag{best_lag}', 'sent_ma5']

X_train_e2e = df_train[E2E_FEATURES]
X_test_e2e  = df_test[E2E_FEATURES]
y_train_e2e = df_train['option_price']
y_test_e2e  = df_test['option_price']

# 线性回归
lr = LinearRegression()
lr.fit(X_train_e2e, y_train_e2e)
mse_lr = mean_squared_error(y_test_e2e, lr.predict(X_test_e2e))
print(f"线性回归 MSE: {mse_lr:.6f}")

# XGBoost
xgb_e2e = XGBRegressor(random_state=42)
xgb_e2e.fit(X_train_e2e, y_train_e2e)
mse_xgb_e2e = mean_squared_error(y_test_e2e, xgb_e2e.predict(X_test_e2e))
print(f"XGBoost MSE: {mse_xgb_e2e:.6f}")

print(f"\n E2E最优模型: {'线性回归' if mse_lr < mse_xgb_e2e else 'XGBoost'}")
print("="*60)
print(" 运行完成！")