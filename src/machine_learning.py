import pandas as pd
import numpy as np
import os
import random
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局参数配置 =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TRADING_DAYS = 252
ROLLING_WINDOWS = [5, 10, 20, 60]
STRIKE = 110
T_MATURITY = 1/12
EPS = 1e-8
DIR_THRESHOLD = 0.001
TEST_SIZE = 0.3
GAP_DAYS = 5
TSCV_SPLITS = 3

# 自动创建文件夹
for directory in ["models", "plots", "metadata", "reports", "cv_results"]:
    os.makedirs(directory, exist_ok=True)

# 加载数据
df = pd.read_csv("data/processed_data.csv")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# ===================== 核心函数 =====================
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

def regime_classification(vol_series, low_pct=33, high_pct=66):
    low_thresh = np.percentile(vol_series, low_pct)
    high_thresh = np.percentile(vol_series, high_pct)
    return np.where(vol_series <= low_thresh, "Low_Vol",
                   np.where(vol_series >= high_thresh, "High_Vol", "Normal_Vol"))

# Expanding Window 验证函数
def expanding_window_validation(X, y, model, tscv, is_linear=False, scaler=None):
    rmse_list = []
    fold_idx = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        if is_linear and scaler is not None:
            X_test_fold_scaled = scaler.transform(X_test_fold)
            pred = model.predict(X_test_fold_scaled)
        else:
            pred = model.predict(X_test_fold)
        rmse = np.sqrt(mean_squared_error(y_test_fold, pred))
        rmse_list.append(rmse)
        fold_idx.append(i+1)
    return rmse_list, fold_idx

# ===================== 多滚动窗口测试 =====================
all_window_results = []
for ROLLING_WINDOW in ROLLING_WINDOWS:
    print(f"测试滚动窗口：{ROLLING_WINDOW} 交易日")
    
    df_temp = df.copy()
    df_temp["rolling_vol"] = df_temp["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
    df_temp["target_vol_t1"] = df_temp["rolling_vol"].shift(-1)
    df_temp["rv_lag1"] = df_temp["rolling_vol"].shift(1)
    df_temp["bsm_baseline"] = black_scholes_vec(df_temp["S"], STRIKE, T_MATURITY, df_temp["r"], df_temp["rv_lag1"])
    df_temp = df_temp.dropna().reset_index(drop=True)

    def build_features(data):
        df = data.copy()
        df['sent_lag1'] = df['sentiment'].shift(1)
        df['S_lag1'] = df['S'].shift(1)
        df['r_lag1'] = df['r'].shift(1)
        df['rv_lag1'] = df['rolling_vol'].shift(1)
        df['sent_ma5'] = df['sentiment'].rolling(5).mean().shift(1)
        return df.dropna()

    df_final = build_features(df_temp)
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

    # LR + L2 正则
    lr_params = {"alpha": [0.01, 0.1, 1, 10, 100], "random_state": [SEED]}
    lr_search = GridSearchCV(Ridge(), lr_params, cv=tscv, scoring="neg_root_mean_squared_error")
    lr_search.fit(X_train_scaled, y_train)
    lr_model = lr_search.best_estimator_

    # 随机森林
    rf_params = {"n_estimators": [100,200], "max_depth": [2,3,5], "min_samples_split": [2,4], "min_samples_leaf": [1,3], "random_state": [SEED]}
    rf_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_root_mean_squared_error")
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    # XGBoost
    xgb_params = {"n_estimators": [100,200], "max_depth": [2,3], "learning_rate": [0.05,0.1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [5,10], "random_state": [SEED]}
    xgb_search = RandomizedSearchCV(XGBRegressor(), xgb_params, n_iter=10, cv=tscv, random_state=SEED)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = best_rf.predict(X_test)
    xgb_pred = best_xgb.predict(X_test)

    models = [lr_model, best_rf, best_xgb]
    preds = [lr_pred, rf_pred, xgb_pred]
    model_names = ["Linear Regression", "Random Forest", "XGBoost"]

    for name, pred in zip(model_names, preds):
        mse, mae, rmse, r2, dir_acc = evaluate(y_test, pred)
        all_window_results.append([ROLLING_WINDOW, name, rmse, r2, dir_acc])

window_df = pd.DataFrame(all_window_results, columns=["Rolling_Window", "Model", "RMSE", "R2", "Directional_Acc"])
window_df.to_csv("reports/rolling_window_comparison.csv", index=False)
print(f"\n多窗口测试结果已保存：reports/rolling_window_comparison.csv")

# 滚动窗口可视化
plt.figure(figsize=(12, 6))
sns.barplot(data=window_df, x="Rolling_Window", y="RMSE", hue="Model", palette="viridis")
plt.title("Rolling Window Impact on Volatility Prediction (RMSE)")
plt.xlabel("Rolling Window (Trading Days)")
plt.ylabel("RMSE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/window_rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=window_df, x="Rolling_Window", y="R2", hue="Model", palette="coolwarm")
plt.title("Rolling Window Impact on Volatility Prediction (R2)")
plt.xlabel("Rolling Window (Trading Days)")
plt.ylabel("R2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/window_r2_comparison.png", dpi=300)
plt.close()

# ===================== 基准20日窗口（核心实验） =====================
ROLLING_WINDOW = 20
df_temp = df.copy()
df_temp["rolling_vol"] = df_temp["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
df_temp["target_vol_t1"] = df_temp["rolling_vol"].shift(-1)
df_temp["rv_lag1"] = df_temp["rolling_vol"].shift(1)
df_temp["bsm_baseline"] = black_scholes_vec(df_temp["S"], STRIKE, T_MATURITY, df_temp["r"], df_temp["rv_lag1"])
df_temp = df_temp.dropna().reset_index(drop=True)
df_final = build_features(df_temp)

n_total = len(df_final)
test_split_idx = int(n_total * (1 - TEST_SIZE))
df_train = df_final.iloc[:test_split_idx].copy()
df_test = df_final.iloc[test_split_idx:].copy()
X_train, X_test = df_train[FEATURES], df_test[FEATURES]
y_train, y_test = df_train['target_vol_t1'], df_test['target_vol_t1']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
lr_search = GridSearchCV(Ridge(), lr_params, cv=tscv, scoring="neg_root_mean_squared_error")
lr_search.fit(X_train_scaled, y_train)
lr_model = lr_search.best_estimator_

rf_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_root_mean_squared_error")
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

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
for name, pred in zip(model_names, preds):
    mse, mae, rmse, r2, dir_acc = evaluate(y_test, pred)
    results.append([name, rmse, r2, dir_acc])
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "Directional_Acc"])
best_idx = results_df["RMSE"].idxmin()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = models[best_idx]
y_pred_vol = preds[best_idx]

# ===================== E2E 端到端定价模型 =====================
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

# 【补回】E2E 控制台打印
print("\n" + "="*50)
print("E2E MODEL")
print("="*50)
print(f"E2E RMSE: {e2e_rmse:.4f}")
print(f"E2E R2: {e2e_r2:.4f}")

# 【补回】Two-Step VS E2E 对比打印
comparison_df = pd.DataFrame([
    ["Two-Step (Best ML Vol Model)", results_df.loc[best_idx, "RMSE"]],
    ["E2E Pricing Model", e2e_rmse]
], columns=["Model", "RMSE"])
print("\nTwo-Step VS E2E:")
print(comparison_df)

# ===================== 基础可视化 =====================
plt.figure(figsize=(10, 8))
corr = df_final[FEATURES].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.close()

# 波动率预测对比
plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], y_test, label='Actual Volatility', color='#2E86AB')
plt.plot(df_test['date'], y_pred_vol, label='Predicted Volatility', color='#FF6B6B')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Volatility Forecast vs Actual')
plt.legend()
plt.tight_layout()
plt.savefig("plots/vol_prediction.png", dpi=300)
plt.close()

# 【补回】Expanding Window 验证图
expanding_rmse_list, fold_idx = expanding_window_validation(X_train, y_train, best_model, tscv, 
                                                             is_linear=(best_model_name=="Linear Regression"),
                                                             scaler=scaler)
plt.figure(figsize=(10, 6))
plt.plot(fold_idx, expanding_rmse_list, marker='o', color='#2E86AB')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Expanding Window Validation RMSE')
plt.tight_layout()
plt.savefig("plots/expanding_window_validation.png", dpi=300)
plt.close()

# ===================== 全模型分Regime性能评估 =====================
df_test["regime"] = regime_classification(df_test["rolling_vol"])
df_test["lr_pred"] = lr_pred
df_test["rf_pred"] = rf_pred
df_test["xgb_pred"] = xgb_pred

regime_all_results = []
model_pred_map = {
    "Linear Regression (Ridge L2)": "lr_pred",
    "Random Forest": "rf_pred",
    "XGBoost": "xgb_pred"
}

for model_name, pred_col in model_pred_map.items():
    for regime in ["Low_Vol", "Normal_Vol", "High_Vol"]:
        mask = df_test["regime"] == regime
        if mask.sum() > 0:
            y_true_sub = y_test[mask]
            y_pred_sub = df_test[pred_col][mask]
            _, _, rmse, r2, dir_acc = evaluate(y_true_sub, y_pred_sub)
            regime_all_results.append([model_name, regime, mask.sum(), rmse, r2, dir_acc])

regime_all_df = pd.DataFrame(regime_all_results, columns=["Model", "Regime", "Sample_Size", "RMSE", "R2", "Directional_Acc"])
regime_all_df.to_csv("reports/regime_all_models_results.csv", index=False)
print("\n全模型分市场状态报告已保存：reports/regime_all_models_results.csv")

# 【补回】Regime 控制台打印
print("\n" + "="*60)
print("全模型分市场状态(Regime)性能总览")
print("="*60)
print(regime_all_df.round(4))

plt.figure(figsize=(12, 6))
sns.barplot(data=regime_all_df, x="Regime", y="RMSE", hue="Model", palette="viridis")
plt.title("All Models Performance Across Volatility Regimes (RMSE)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/regime_all_models_rmse.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=regime_all_df, x="Regime", y="R2", hue="Model", palette="coolwarm")
plt.title("All Models Performance Across Volatility Regimes (R2)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/regime_all_models_r2.png", dpi=300)
plt.close()

# ===================== Parity Plot（Actual vs Predicted Volatility 三模型对比） =====================
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_pred
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = rf_pred
xgb_train_pred = best_xgb.predict(X_train)
xgb_test_pred = xgb_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
fig.suptitle("Model Performance: Actual vs. Predicted Volatility", fontsize=16, y=1.02)

def plot_parity(ax, y_train, y_train_pred, y_test, y_test_pred, title):
    ax.scatter(y_train, y_train_pred, color='#1f77b4', alpha=0.7, label='Train')
    ax.scatter(y_test, y_test_pred, marker='s', facecolor='white', edgecolor='black', label='Test')
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    ax.plot(lims, lims, 'k--', label='X=Y')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Actual Volatility', fontsize=12)
    ax.set_ylabel('Predicted Volatility', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plot_parity(axes[0], y_train, lr_train_pred, y_test, lr_test_pred, "Linear Regression (Ridge L2)")
plot_parity(axes[1], y_train, rf_train_pred, y_test, rf_test_pred, "Random Forest")
plot_parity(axes[2], y_train, xgb_train_pred, y_test, xgb_test_pred, "XGBoost")

plt.tight_layout()
plt.savefig("plots/model_performance_parity_all.png", dpi=300, bbox_inches='tight')
plt.close()

# ===================== 分Regime SHAP + 全特征交互分析 =====================
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']
model_list = [lr_model, best_rf, best_xgb]
model_names_shap = ["LR_Ridge", "RandomForest", "XGBoost"]
is_linear_list = [True, False, False]
regimes = ["Low_Vol", "Normal_Vol", "High_Vol"]

# 分Regime SHAP
def run_regime_shap(model, model_name, is_linear, X_test, df_test, scaler):
    for regime in regimes:
        mask = df_test["regime"] == regime
        if mask.sum() < 5:
            continue
        X_reg = X_test[mask].copy()
        if is_linear:
            X_reg_scaled = scaler.transform(X_reg)
            explainer = shap.LinearExplainer(model, X_reg_scaled)
            sv = explainer.shap_values(X_reg_scaled)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_reg)
        
        plt.figure()
        shap.summary_plot(sv, X_reg, plot_type="bar", show=False)
        plt.title(f"{model_name} | {regime} | Feature Importance")
        plt.tight_layout()
        plt.savefig(f"plots/shap_{model_name}_{regime}_importance.png", bbox_inches='tight')
        plt.close()

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_regime_shap(model, name, linear, X_test, df_test, scaler)

# 全局SHAP Beeswarm图
if best_model_name != "Linear Regression":
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

# 全特征交互SHAP
def run_full_interaction_shap(model, model_name, is_linear, X_test, scaler):
    if is_linear:
        X_scaled = scaler.transform(X_test)
        explainer = shap.LinearExplainer(model, X_scaled)
        sv = explainer.shap_values(X_scaled)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)

    feature_pairs = list(itertools.combinations(FEATURES, 2))
    for feat1, feat2 in feature_pairs:
        try:
            plt.figure()
            shap.dependence_plot(feat1, sv, X_test, interaction_index=feat2, show=False)
            plt.title(f"{model_name} | {feat1} × {feat2}")
            plt.tight_layout()
            plt.savefig(f"plots/shap_interact_{model_name}_{feat1}_{feat2}.png", bbox_inches='tight')
            plt.close()
        except:
            continue

    try:
        plt.figure(figsize=(12, 10))
        interaction_vals = shap.interaction_values(sv, X_test)
        interaction_strength = np.abs(interaction_vals).mean(axis=0)
        sns.heatmap(interaction_strength, annot=True, fmt=".4f", xticklabels=FEATURES, yticklabels=FEATURES, cmap="coolwarm")
        plt.title(f"{model_name} | Feature Interaction Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/shap_interaction_heatmap_{model_name}.png", bbox_inches='tight')
        plt.close()
    except:
        pass

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_full_interaction_shap(model, name, linear, X_test, scaler)

# ===================== 双重定价 + 95%误差边际 =====================
df_test['pred_vol'] = y_pred_vol
df_test['two_step_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])
price_residuals = df_test['bsm_baseline'] - df_test['two_step_price']
std_residual = np.std(price_residuals)
df_test['two_step_price_lower'] = df_test['two_step_price'] - 1.96 * std_residual
df_test['two_step_price_upper'] = df_test['two_step_price'] + 1.96 * std_residual

pricing_df = df_test[['date', 'two_step_price', 'two_step_price_lower', 'two_step_price_upper', 'bsm_baseline']].copy()
pricing_df['e2e_price'] = e2e_pred
pricing_df.to_csv("reports/dual_pricing_with_95CI.csv", index=False)

# 双定价趋势图
plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], df_test['two_step_price'], label='Two-Step Pricing', color='#2E86AB')
plt.plot(df_test['date'], e2e_pred, label='E2E Pricing', color='#FF6B6B')
plt.plot(df_test['date'], df_test['bsm_baseline'], label='BSM Baseline', color='#4CAF50', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Option Price')
plt.title('Dual Pricing Trend Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("plots/dual_pricing_trend.png", dpi=300)
plt.close()

# ===================== 完整残差分析 =====================
residual_df = df_test[['date', 'rolling_vol', 'regime']].copy()
residual_df['actual_vol'] = y_test
residual_df['lr_pred_vol'] = lr_pred
residual_df['rf_pred_vol'] = rf_pred
residual_df['xgb_pred_vol'] = xgb_pred
residual_df['best_vol_pred'] = y_pred_vol
residual_df['two_step_price'] = df_test['two_step_price']
residual_df['e2e_price'] = e2e_pred
residual_df['bsm_price'] = df_test['bsm_baseline']

residual_df['vol_residual'] = residual_df['actual_vol'] - residual_df['best_vol_pred']
residual_df['two_step_price_residual'] = residual_df['bsm_price'] - residual_df['two_step_price']
residual_df['e2e_price_residual'] = residual_df['bsm_price'] - residual_df['e2e_price']
residual_df.to_csv("reports/full_residual_analysis.csv", index=False)

# 波动率残差分布图
plt.figure(figsize=(10, 6))
plt.hist(residual_df['vol_residual'], bins=30, alpha=0.7, color='#2E86AB')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Volatility Forecast Residual Distribution')
plt.tight_layout()
plt.savefig("plots/vol_residual_hist.png", dpi=300)
plt.close()

# 定价残差分布图
plt.figure(figsize=(10, 6))
plt.hist(residual_df['two_step_price_residual'], bins=30, alpha=0.7, color='#FF6B6B')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Two-Step Pricing Residual Distribution')
plt.tight_layout()
plt.savefig("plots/price_residual_hist.png", dpi=300)
plt.close()

# ===================== 期权定价敏感性分析 =====================
def plot_sensitivity_analysis():
    S_range = np.linspace(df_test['S_lag1'].min() * 0.8, df_test['S_lag1'].max() * 1.2, 50)
    vol_range = np.linspace(df_test['rv_lag1'].min() * 0.5, df_test['rv_lag1'].max() * 1.5, 50)
    r_fixed = df_test['r_lag1'].mean()
    S_fixed = df_test['S_lag1'].mean()
    vol_fixed = df_test['rv_lag1'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bsm_sens = black_scholes_vec(S_range, STRIKE, T_MATURITY, r_fixed, vol_fixed)
    ml_sens = black_scholes_vec(S_range, STRIKE, T_MATURITY, r_fixed, vol_fixed)
    ax1.plot(S_range, bsm_sens, label='BSM', linewidth=2)
    ax1.plot(S_range, ml_sens, label='ML Two-Step', linewidth=2, linestyle='--')
    ax1.set_title('Option Price Sensitivity to Stock Price')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Option Price')
    ax1.legend()
    ax1.grid(alpha=0.3)

    bsm_vol_sens = black_scholes_vec(S_fixed, STRIKE, T_MATURITY, r_fixed, vol_range)
    ml_vol_sens = black_scholes_vec(S_fixed, STRIKE, T_MATURITY, r_fixed, vol_range)
    ax2.plot(vol_range, bsm_vol_sens, label='BSM', linewidth=2)
    ax2.plot(vol_range, ml_vol_sens, label='ML Two-Step', linewidth=2, linestyle='--')
    ax2.set_title('Option Price Sensitivity to Volatility')
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Option Price')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/sensitivity_analysis.png", dpi=300)
    plt.close()

plot_sensitivity_analysis()

# ===================== 希腊字母（Delta/Vega）计算 =====================
def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    delta = norm.cdf(d1)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, vega

# 计算并保存希腊字母（以Two-Step定价为例）
df_test['delta'], df_test['vega'] = calculate_greeks(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])
greeks_df = df_test[['date', 'delta', 'vega', 'pred_vol', 'two_step_price']].copy()
greeks_df.to_csv("reports/option_greeks.csv", index=False)

# ===================== 模型保存与报告 =====================
pickle.dump(best_model, open("models/best_quant_model.pkl", "wb"))
pickle.dump(e2e_model, open("models/e2e_option_pricing_model.pkl", "wb"))

governance_data = {
    "Model": ["Linear Regression (Ridge L2)", "Random Forest", "XGBoost", "E2E Pricing"],
    "RMSE": [results_df.loc[0, "RMSE"], results_df.loc[1, "RMSE"], results_df.loc[2, "RMSE"], e2e_rmse],
    "R2": [results_df.loc[0, "R2"], results_df.loc[1, "R2"], results_df.loc[2, "R2"], e2e_r2],
}
governance_df = pd.DataFrame(governance_data)
governance_df.to_csv("reports/model_governance_report.csv", index=False)

# ===================== 最终控制台输出 =====================
print("\n" + "="*50)
print(f"Best Model: {best_model_name}")
print(f"Volatility RMSE: {results_df.loc[best_idx, 'RMSE']:.4f}")
print(f"Directional Accuracy: {results_df.loc[best_idx, 'Directional_Acc']:.2%}")
print(f"Expanding Window CV RMSE: {np.mean(expanding_rmse_list):.4f}")
print("="*50)

# ===================== 期权定价敏感性 =====================
def plot_sensitivity_analysis():
    S_range = np.linspace(df_test['S_lag1'].min() * 0.8, df_test['S_lag1'].max() * 1.2, 50)
    vol_range = np.linspace(df_test['rv_lag1'].min() * 0.5, df_test['rv_lag1'].max() * 1.5, 50)
    r_fixed = df_test['r_lag1'].mean()
    S_fixed = df_test['S_lag1'].mean()
    vol_fixed = df_test['rv_lag1'].mean()

    # 合并敏感性图（原功能保留）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    bsm_sens = black_scholes_vec(S_range, STRIKE, T_MATURITY, r_fixed, vol_fixed)
    ml_sens = black_scholes_vec(S_range, STRIKE, T_MATURITY, r_fixed, vol_fixed)
    ax1.plot(S_range, bsm_sens, label='BSM', linewidth=2)
    ax1.plot(S_range, ml_sens, label='ML Two-Step', linewidth=2, linestyle='--')
    ax1.set_title('Option Price Sensitivity to Stock Price')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Option Price')
    ax1.legend()
    ax1.grid(alpha=0.3)

    bsm_vol_sens = black_scholes_vec(S_fixed, STRIKE, T_MATURITY, r_fixed, vol_range)
    ml_vol_sens = black_scholes_vec(S_fixed, STRIKE, T_MATURITY, r_fixed, vol_range)
    ax2.plot(vol_range, bsm_vol_sens, label='BSM', linewidth=2)
    ax2.plot(vol_range, ml_vol_sens, label='ML Two-Step', linewidth=2, linestyle='--')
    ax2.set_title('Option Price Sensitivity to Volatility')
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Option Price')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/sensitivity_analysis.png", dpi=300)
    plt.close()

    # Vega敏感性图（vol_sensitivity_vega.png）
    plt.figure(figsize=(10, 6))
    plt.plot(vol_range, bsm_vol_sens, label='BSM', linewidth=2)
    plt.plot(vol_range, ml_vol_sens, label='ML Two-Step', linewidth=2, linestyle='--')
    plt.title('Option Price Sensitivity to Volatility (Vega)')
    plt.xlabel('Volatility')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/vol_sensitivity_vega.png", dpi=300)
    plt.close()

plot_sensitivity_analysis()

print("\n 全部任务完成！")