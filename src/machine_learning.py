import pandas as pd
import numpy as np
import os
import random
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TRADING_DAYS = 252
ROLLING_WINDOW = 20
STRIKE = 110
T_MATURITY = 1/12
EPS = 1e-8
DIR_THRESHOLD = 0.001
TEST_SIZE = 0.3
GAP_DAYS = 5
TSCV_SPLITS = 3

# 创建所有文件夹
for directory in ["models", "plots", "metadata", "reports", "cv_results"]:
    os.makedirs(directory, exist_ok=True)

df = pd.read_csv("data/processed_data.csv")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
df["target_vol_t1"] = df["rolling_vol"].shift(-1)
df["rv_lag1"] = df["rolling_vol"].shift(1)

def black_scholes_vec(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    invalid = (sigma <= 1e-6) | (T <= 0) | np.isnan(S) | np.isnan(r) | np.isnan(sigma)
    sigma_safe = np.where(invalid, 1.0, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * np.sqrt(T))
    d2 = d1 - sigma_safe * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(invalid, np.nan, price)

df["bsm_baseline"] = black_scholes_vec(df["S"], STRIKE, T_MATURITY, df["r"], df["rv_lag1"])
df = df.dropna().reset_index(drop=True)

def build_features(data):
    df = data.copy()
    df['sent_lag1'] = df['sentiment'].shift(1)
    df['S_lag1'] = df['S'].shift(1)
    df['r_lag1'] = df['r'].shift(1)
    df['rv_lag1'] = df['rolling_vol'].shift(1)
    df['sent_ma5'] = df['sentiment'].rolling(5).mean().shift(1)
    return df.dropna()

df_final = build_features(df)
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']

n_total = len(df_final)
test_split_idx = int(n_total * (1 - TEST_SIZE))
df_train = df_final.iloc[:test_split_idx].copy()
df_test = df_final.iloc[test_split_idx:].copy()

X_train, X_test = df_train[FEATURES], df_test[FEATURES]
y_train, y_test = df_train['target_vol_t1'], df_test['target_vol_t1']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

try:
    tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS, gap=GAP_DAYS)
except TypeError:
    tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    dir_acc = 0.0
    if len(y_true) >= 2:
        true_diff = y_true[1:] - y_true[:-1]
        pred_diff = y_pred[1:] - y_pred[:-1]
        true_dir = np.sign(true_diff)
        pred_dir = np.sign(pred_diff)
        valid_mask = np.abs(true_diff) > DIR_THRESHOLD
        if valid_mask.sum() > 0:
            dir_acc = np.mean(true_dir[valid_mask] == pred_dir[valid_mask])
    
    return mse, mae, rmse, r2, dir_acc

# ===================== 1. 两步式波动率模型训练 =====================
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_params = {
    "n_estimators": [100,200], "max_depth": [2,3,5],
    "min_samples_split": [2,4], "min_samples_leaf": [1,3],
    "random_state": [SEED]
}
rf_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_root_mean_squared_error")
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

xgb_params = {
    "n_estimators": [100,200], "max_depth": [2,3], "learning_rate": [0.05,0.1],
    "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [5,10],
    "random_state": [SEED]
}
xgb_search = RandomizedSearchCV(XGBRegressor(), xgb_params, n_iter=10, cv=tscv, random_state=SEED)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

lr_pred = lr_model.predict(X_test_scaled)
rf_pred = best_rf.predict(X_test)
xgb_pred = best_xgb.predict(X_test)

models = [lr_model, best_rf, best_xgb]
preds = [lr_pred, rf_pred, xgb_pred]
model_names = ["Linear Regression", "Random Forest", "XGBoost"]
results = []

# 计算Persistence Baseline
persistence_pred = df_test['rv_lag1']
per_mse, per_mae, per_rmse, per_r2, _ = evaluate(y_test, persistence_pred)

# 生成性能表格
performance_data = []
for name, pred in zip(model_names, preds):
    mse, mae, rmse, r2, dir_acc = evaluate(y_test, pred)
    results.append([name, rmse, r2, dir_acc])
    performance_data.append([name, mae, rmse, r2])
performance_data.append(["Persistence Baseline", per_mae, per_rmse, per_r2])

vol_perf_df = pd.DataFrame(performance_data, columns=["Model", "MAE", "RMSE", "R2"])
csv_path = "/Users/yoyo/Desktop/MS_CHOOSER_OPTION/cv_results/volatility_performance.csv"
vol_perf_df.to_csv(csv_path, index=False)
print(f"波动率性能表格已保存至：{csv_path}")

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "Directional_Acc"])
best_idx = results_df["RMSE"].idxmin()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = models[best_idx]
y_pred_vol = preds[best_idx]

# ===================== 2. E2E 端到端模型 =====================
E2E_FEATURES = ['S_lag1','r_lag1','rv_lag1','sent_lag1','sent_ma5']
df_final["target_price"] = df_final["bsm_baseline"]
X_e2e = df_final[E2E_FEATURES]
y_e2e = df_final["target_price"]
X_train_e2e = X_e2e.iloc[:test_split_idx]
X_test_e2e = X_e2e.iloc[test_split_idx:]
y_train_e2e = y_e2e.iloc[:test_split_idx]
y_test_e2e = y_e2e.iloc[test_split_idx:]

scaler_e2e = StandardScaler()
X_train_e2e_scaled = scaler_e2e.fit_transform(X_train_e2e)
X_test_e2e_scaled = scaler_e2e.transform(X_test_e2e)

e2e_model = XGBRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=5,random_state=SEED)
e2e_model.fit(X_train_e2e_scaled, y_train_e2e)
e2e_pred = e2e_model.predict(X_test_e2e_scaled)
e2e_mse, e2e_mae, e2e_rmse, e2e_r2, e2e_dir = evaluate(y_test_e2e, e2e_pred)

print("\n" + "="*50)
print("E2E MODEL")
print("="*50)
print(f"E2E RMSE: {e2e_rmse:.4f}")
print(f"E2E R2: {e2e_r2:.4f}")

comparison_df = pd.DataFrame([
    ["Two-Step (Best ML Vol Model)", results_df.loc[best_idx, "RMSE"]],
    ["E2E Pricing Model", e2e_rmse]
], columns=["Model", "RMSE"])
print("\nTwo-Step VS E2E:")
print(comparison_df)

# 3.1 特征相关性热力图
plt.figure(figsize=(10, 8))
corr = df_final[FEATURES].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.2 滚动窗口验证曲线
def expanding_window_validation(X, y, model, tscv, is_linear=False):
    rmse_list = []
    fold_idx = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        if is_linear:
            X_test_fold_scaled = scaler.transform(X_test_fold)
            pred = model.predict(X_test_fold_scaled)
        else:
            pred = model.predict(X_test_fold)
        rmse = np.sqrt(mean_squared_error(y_test_fold, pred))
        rmse_list.append(rmse)
        fold_idx.append(i+1)
    return rmse_list, fold_idx

expanding_rmse_list, fold_idx = expanding_window_validation(X_train, y_train, best_model, tscv, is_linear=(best_model_name=="Linear Regression"))
plt.figure(figsize=(10, 6))
plt.plot(fold_idx, expanding_rmse_list, marker='o', color='#2E86AB')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Expanding Window Validation RMSE')
plt.tight_layout()
plt.savefig("plots/expanding_window_validation.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.3 波动率预测vs实际对比
plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], y_test, label='Actual Volatility', color='#2E86AB')
plt.plot(df_test['date'], y_pred_vol, label='Predicted Volatility', color='#FF6B6B')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Volatility Forecast vs Actual')
plt.legend()
plt.tight_layout()
plt.savefig("plots/vol_prediction.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.4 波动率残差分布
vol_residuals = y_test - y_pred_vol
plt.figure(figsize=(10, 6))
plt.hist(vol_residuals, bins=30, color='#2E86AB', alpha=0.7)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Volatility Forecast Residual Distribution')
plt.tight_layout()
plt.savefig("plots/vol_residual_hist.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.5 期权定价残差分布
price_residuals = y_test_e2e - e2e_pred
plt.figure(figsize=(10, 6))
plt.hist(price_residuals, bins=30, color='#FF6B6B', alpha=0.7)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Option Pricing Residual Distribution')
plt.tight_layout()
plt.savefig("plots/price_residual_hist.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.6 双定价趋势对比
df_test['pred_vol'] = y_pred_vol
df_test['two_step_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])

plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], df_test['two_step_price'], label='Two-Step Pricing', color='#2E86AB')
plt.plot(df_test['date'], e2e_pred, label='E2E Pricing', color='#FF6B6B')
plt.plot(df_test['date'], df_test['bsm_baseline'], label='BSM Baseline', color='#4CAF50', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Option Price')
plt.title('Dual Pricing Trend Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("plots/dual_pricing_trend.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.7 SHAP Beeswarm 图
if best_model_name != "Linear Regression":
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3.8 E2E SHAP 图
explainer_e2e = shap.TreeExplainer(e2e_model)
shap_values_e2e = explainer_e2e.shap_values(X_test_e2e_scaled)
shap.summary_plot(shap_values_e2e, X_test_e2e, plot_type="bar", show=False)
plt.savefig("plots/shap_e2e.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.9 模型性能Parity Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_vol, alpha=0.6, color='#2E86AB')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('Model Performance Parity Plot')
plt.tight_layout()
plt.savefig("plots/model_performance_parity_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.10 波动率预测仪表盘
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].plot(df_test['date'], y_test, label='Actual')
axes[0,0].plot(df_test['date'], y_pred_vol, label='Predicted')
axes[0,0].set_title('Volatility Forecast')
axes[0,1].scatter(y_test, y_pred_vol)
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0,1].set_title('Parity Plot')
axes[1,0].hist(vol_residuals, bins=30)
axes[1,0].set_title('Residual Distribution')
axes[1,1].bar(results_df['Model'], results_df['RMSE'])
axes[1,1].set_title('Model RMSE Comparison')
plt.tight_layout()
plt.savefig("plots/vol_forecast_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

# SHAP 重要性图
if best_model_name == "Linear Regression":
    explainer = shap.LinearExplainer(best_model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("plots/shap_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 分市场状态测试
def regime_classification(vol_series, low_pct=33, high_pct=66):
    low_thresh = np.percentile(vol_series, low_pct)
    high_thresh = np.percentile(vol_series, high_pct)
    return np.where(vol_series <= low_thresh, "Low_Vol",
                   np.where(vol_series >= high_thresh, "High_Vol", "Normal_Vol"))

df_test["regime"] = regime_classification(df_test["rolling_vol"])
regime_results = []
for regime in ["Low_Vol", "Normal_Vol", "High_Vol"]:
    mask = df_test["regime"] == regime
    if mask.sum() > 0:
        _, _, rmse, r2, _ = evaluate(y_test[mask], y_pred_vol[mask])
        regime_results.append([regime, mask.sum(), rmse, r2])

regime_df = pd.DataFrame(regime_results, columns=["Regime", "Sample_Size", "RMSE", "R2"])
regime_df.to_csv("reports/regime_test_results.csv", index=False)

# 打印最终结果
print("\n" + "="*50)
print(f"Best Model: {best_model_name}")
print(f"Volatility RMSE: {results_df.loc[best_idx, 'RMSE']:.4f}")
print(f"Directional Accuracy: {results_df.loc[best_idx, 'Directional_Acc']:.2%}")
print(f"Expanding Window CV RMSE: {np.mean(expanding_rmse_list):.4f}")
print("="*50)

# 保存模型
pickle.dump(best_model, open("models/best_quant_model.pkl", "wb"))
pickle.dump(e2e_model, open("models/e2e_option_pricing_model.pkl", "wb"))

# 模型性能对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_vol, alpha=0.6, color='#2E86AB')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('Actual vs Predicted Volatility')
plt.tight_layout()
plt.savefig("plots/model_performance_actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.close()

# ===================== 三模型Actual vs Predicted Volatility对比图 =====================
# Linear Regression
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_pred

# Random Forest
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = rf_pred

# XGBoost
xgb_train_pred = best_xgb.predict(X_train)
xgb_test_pred = xgb_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
fig.suptitle("Model Performance: Actual vs. Predicted Volatility", fontsize=16, y=1.02)

def plot_parity(ax, y_train, y_train_pred, y_test, y_test_pred, title):
    ax.scatter(y_train, y_train_pred, color='#1f77b4', alpha=0.7, label='Train')
    ax.scatter(y_test, y_test_pred, marker='s', facecolor='white', edgecolor='black', label='Test')
    # X=Y对角线
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    ax.plot(lims, lims, 'k--', label='X=Y')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Actual Volatility', fontsize=12)
    ax.set_ylabel('Predicted Volatility', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

# 绘制三个模型的对比
plot_parity(axes[0], y_train, lr_train_pred, y_test, lr_test_pred, "Linear Regression")
plot_parity(axes[1], y_train, rf_train_pred, y_test, rf_test_pred, "Random Forest")
plot_parity(axes[2], y_train, xgb_train_pred, y_test, xgb_test_pred, "XGBoost")

plt.tight_layout()
plt.savefig("plots/model_performance_parity_all.png", dpi=300, bbox_inches='tight')
plt.close()


# --------------------- 1. regime_test_results.csv ---------------------
def regime_classification(vol_series, low_pct=33, high_pct=66):
    low_thresh = np.percentile(vol_series, low_pct)
    high_thresh = np.percentile(vol_series, high_pct)
    return np.where(vol_series <= low_thresh, "Low_Vol",
                   np.where(vol_series >= high_thresh, "High_Vol", "Normal_Vol"))

df_test["regime"] = regime_classification(df_test["rolling_vol"])
regime_results = []
for regime in ["Low_Vol", "Normal_Vol", "High_Vol"]:
    mask = df_test["regime"] == regime
    if mask.sum() > 0:
        _, _, rmse, r2, _ = evaluate(y_test[mask], y_pred_vol[mask])
        regime_results.append([regime, mask.sum(), rmse, r2])

regime_df = pd.DataFrame(regime_results, columns=["Regime", "Sample_Size", "RMSE", "R2"])
regime_df.to_csv("reports/regime_test_results.csv", index=False)

# --------------------- 2. dual_pricing_with_95CI.csv ---------------------
df_test['pred_vol'] = y_pred_vol
df_test['two_step_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])

price_residuals = df_test['bsm_baseline'] - df_test['two_step_price']
std_residual = np.std(price_residuals)
df_test['two_step_price_lower'] = df_test['two_step_price'] - 1.96 * std_residual
df_test['two_step_price_upper'] = df_test['two_step_price'] + 1.96 * std_residual

# 合并E2E定价结果
pricing_df = df_test[['date', 'two_step_price', 'two_step_price_lower', 'two_step_price_upper', 'bsm_baseline']].copy()
pricing_df['e2e_price'] = e2e_pred
pricing_df.to_csv("reports/dual_pricing_with_95CI.csv", index=False)

# --------------------- 3. model_governance_report.csv ---------------------
governance_data = {
    "Model": ["Linear Regression", "Random Forest", "XGBoost", "E2E Pricing"],
    "RMSE": [results_df.loc[0, "RMSE"], results_df.loc[1, "RMSE"], results_df.loc[2, "RMSE"], e2e_rmse],
    "R2": [results_df.loc[0, "R2"], results_df.loc[1, "R2"], results_df.loc[2, "R2"], e2e_r2],
    "Random_State": [SEED, SEED, SEED, SEED],
    "Train_Test_Split": [f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}"]*4,
    "TSCV_Splits": [TSCV_SPLITS]*4,
    "GAP_DAYS": [GAP_DAYS]*4,
    "E2E_Hyperparameters": ["-", "-", "-", "n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5"]
}
governance_df = pd.DataFrame(governance_data)
governance_df.to_csv("reports/model_governance_report.csv", index=False)

print("\n 图表、报告、模型已生成")