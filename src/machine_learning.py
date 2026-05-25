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
DIR_THRESHOLD = 1e-4 
TEST_SIZE = 0.5
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
        valid_mask = (true_dir != 0) & (pred_dir != 0)
        if valid_mask.sum() > 0:
            dir_acc = np.mean(true_dir[valid_mask] == pred_dir[valid_mask])
    
    return mse, mae, rmse, r2, dir_acc

def regime_classification(vol_series, mid_pct=50):
    mid_thresh = np.percentile(vol_series, mid_pct)
    return np.where(vol_series <= mid_thresh, "Low_Vol", "High_Vol")

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

# ===================== 基准20日窗口 =====================
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
df_final["true_bsm_price"] = black_scholes_vec(df_final["S"], STRIKE, T_MATURITY, df_final["r"], df_final["target_vol_t1"])
X_e2e = df_final[E2E_FEATURES]
y_e2e = df_final["true_bsm_price"] 
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

# E2E 控制台打印
print("\n" + "="*50)
print("E2E期权定价（R²高为正常，因拟合期权价格）")
print("="*50)
print(f"E2E RMSE: {e2e_rmse:.4f}")
print(f"E2E R2: {e2e_r2:.4f}")

# ===================== 基础可视化 =====================
plt.figure(figsize=(10, 8))
corr = df_final[FEATURES].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap (Volatility Prediction)")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.close()

# 波动率预测对比
plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], y_test, label='Actual Volatility', color='#2E86AB')
plt.plot(df_test['date'], y_pred_vol, label='Predicted Volatility', color='#FF6B6B')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Volatility Forecast vs Actual (Volatility Prediction)')
plt.legend()
plt.tight_layout()
plt.savefig("plots/vol_prediction.png", dpi=300)
plt.close()

# Expanding Window 验证图
expanding_rmse_list, fold_idx = expanding_window_validation(X_train, y_train, best_model, tscv, 
                                                             is_linear=(best_model_name=="Linear Regression"),
                                                             scaler=scaler)
plt.figure(figsize=(10, 6))
plt.plot(fold_idx, expanding_rmse_list, marker='o', color='#2E86AB')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Expanding Window Validation RMSE (Volatility Prediction)')
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
    for regime in ["Low_Vol", "High_Vol"]:
        mask = df_test["regime"] == regime
        if mask.sum() > 0:
            y_true_sub = y_test[mask]
            y_pred_sub = df_test[pred_col][mask]
            _, _, rmse, r2, dir_acc = evaluate(y_true_sub, y_pred_sub)
            regime_all_results.append([model_name, regime, mask.sum(), rmse, r2, dir_acc])

# 加入BSM波动率基准
baseline_map = {
    "BSM Baseline (Lagged Vol)": "rv_lag1"
}
for model_name, pred_col in baseline_map.items():
    for regime in ["Low_Vol", "High_Vol"]:
        mask = df_test["regime"] == regime
        if mask.sum() > 0:
            y_true_sub = y_test[mask]
            y_pred_sub = df_test[pred_col][mask]
            _, _, rmse, r2, dir_acc = evaluate(y_true_sub, y_pred_sub)
            regime_all_results.append([model_name, regime, mask.sum(), rmse, r2, dir_acc])

regime_all_df = pd.DataFrame(regime_all_results, columns=["Model", "Regime", "Sample_Size", "RMSE", "R2", "Directional_Acc"])
regime_all_df.to_csv("reports/regime_all_models_results.csv", index=False)
print("\n全模型分市场状态报告已保存：reports/regime_all_models_results.csv")

# Regime 控制台打印
print("\n" + "="*60)
print("全模型分市场状态(Regime)性能总览（Volatility Prediction）")
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

# ===================== Parity Plot（Volatility Prediction） =====================
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

# ===================== SHAP分析 =====================
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']
model_list = [lr_model, best_rf, best_xgb]
model_names_shap = ["LR_Ridge", "RandomForest", "XGBoost"]
is_linear_list = [True, False, False]
regimes = ["Low_Vol", "High_Vol"]

# ===================== 线性回归标准化回归系数 =====================
def plot_standardized_coefficients(model, features):
    # 特征已标准化，模型系数 = 标准化回归系数
    coef = model.coef_
    coef_df = pd.DataFrame({
        "Feature": features,
        "Standardized_Coefficient": coef
    })
    # 按系数绝对值排序，方便查看重要性
    coef_df["Abs_Coeff"] = coef_df["Standardized_Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coeff", ascending=False).reset_index(drop=True)
    
    # 保存系数表格
    coef_df.to_csv("reports/standardized_coefficients.csv", index=False)
    # 控制台打印
    print("\n" + "="*65)
    print("Linear Regression (Ridge) 标准化回归系数")
    print("="*65)
    print(coef_df[["Feature", "Standardized_Coefficient"]].round(4))
    
    # 绘制系数条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Standardized_Coefficient", y="Feature", data=coef_df, palette="coolwarm")
    plt.title("Ridge Model - Standardized Regression Coefficients", fontsize=14)
    plt.xlabel("Standardized Coefficient (Feature Impact Size)")
    plt.ylabel("Input Feature")
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("plots/standardized_coefficients.png", dpi=300)
    plt.close()
    return coef_df

# 执行计算（仅线性回归需要）
standardized_coef_df = plot_standardized_coefficients(lr_model, FEATURES)

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
        plt.title(f"{model_name} | {regime} | Feature Importance (Volatility)")
        plt.tight_layout()
        plt.savefig(f"plots/shap_{model_name}_{regime}_importance.png", bbox_inches='tight')
        plt.close()

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_regime_shap(model, name, linear, X_test, df_test, scaler)

if best_model_name != "Linear Regression":
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

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
            plt.title(f"{model_name} | {feat1} × {feat2} (Volatility)")
            plt.tight_layout()
            plt.savefig(f"plots/shap_interact_{model_name}_{feat1}_{feat2}.png", bbox_inches='tight')
            plt.close()
        except:
            continue

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_full_interaction_shap(model, name, linear, X_test, scaler)

# ===================== 双重定价 =====================
df_test['pred_vol'] = y_pred_vol
df_test['two_step_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])
price_residuals = df_test['bsm_baseline'] - df_test['two_step_price']
std_residual = np.std(price_residuals)
df_test['two_step_price_lower'] = df_test['two_step_price'] - 1.96 * std_residual
df_test['two_step_price_upper'] = df_test['two_step_price'] + 1.96 * std_residual

pricing_df = df_test[['date', 'two_step_price', 'two_step_price_lower', 'two_step_price_upper', 'bsm_baseline']].copy()
pricing_df['e2e_price'] = e2e_pred
pricing_df.to_csv("reports/dual_pricing_with_95CI.csv", index=False)

# ===================== 残差分析 =====================
residual_df = df_test[['date', 'rolling_vol', 'regime']].copy()
residual_df['actual_vol'] = y_test
residual_df['lr_pred_vol'] = lr_pred
residual_df['rf_pred_vol'] = rf_pred
residual_df['xgb_pred_vol'] = xgb_pred
residual_df['best_vol_pred'] = y_pred_vol
residual_df['two_step_price'] = df_test['two_step_price']
residual_df['e2e_price'] = e2e_pred
residual_df['true_bsm_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], y_test)
residual_df['bsm_baseline_price'] = df_test['bsm_baseline'] 

residual_df['vol_residual'] = residual_df['actual_vol'] - residual_df['best_vol_pred']
residual_df['two_step_price_residual'] = residual_df['true_bsm_price'] - residual_df['two_step_price']
residual_df['e2e_price_residual'] = residual_df['true_bsm_price'] - residual_df['e2e_price']
residual_df.to_csv("reports/full_residual_analysis.csv", index=False)

plt.figure(figsize=(10, 6))
plt.hist(residual_df['vol_residual'], bins=30, alpha=0.7, color='#2E86AB')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Volatility Forecast Residual Distribution')
plt.tight_layout()
plt.savefig("plots/vol_residual_hist.png", dpi=300)
plt.close()

# ===================== 敏感性分析 =====================
def plot_final_uncertainty_sensitivity():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    plt.rcParams['axes.unicode_minus'] = False

    # ===================== 1. 全局标准化配置 =====================
    # 训练集OOD安全裁剪阈值（1%/99%分位数）
    sent_lag1_lower = df_train['sent_lag1'].quantile(0.01)
    sent_lag1_upper = df_train['sent_lag1'].quantile(0.99)
    sent_ma5_lower = df_train['sent_ma5'].quantile(0.01)
    sent_ma5_upper = df_train['sent_ma5'].quantile(0.99)

    # 模型配置
    model_info = {
        "Ridge": {"model": lr_model, "linear": True, "color": "#79c0f2"},
        "RF": {"model": best_rf, "linear": False, "color": "#5BE1A0"},
        "XGB": {"model": best_xgb, "linear": False, "color": "#bd6aaa"}
    }
    N_POINTS = 100
    VOL_FLOOR = 1e-4         
    X_THRESHOLD = 0.05    
    OFFSET_RANGE = np.linspace(-0.5, 0.5, N_POINTS)  

    # ===================== 2. 分Regime数据配置 =====================
    df_low = df_test[df_test['regime'] == 'Low_Vol'].copy()
    df_high = df_test[df_test['regime'] == 'High_Vol'].copy()

    regime_cfg = {
        "Low_Vol": {
            "df": df_low,
            "S": df_low['S_lag1'].median(),
            "r": df_low['r_lag1'].median(),
            "rv": df_low['rv_lag1'].median()
        },
        "High_Vol": {
            "df": df_high,
            "S": df_high['S_lag1'].median(),
            "r": df_high['r_lag1'].median(),
            "rv": df_high['rv_lag1'].median()
        },
    }

    # ===================== 3. 遍历市场状态 =====================
    for regime, cfg in regime_cfg.items():
        df_reg = cfg["df"]
        # 安全赋值
        S_fix, r_fix, rv_fix = cfg["S"], cfg["r"], cfg["rv"]
        
        # 情绪变量统计量
        lag1_mu, lag1_std = df_reg["sent_lag1"].mean(), df_reg["sent_lag1"].std()
        ma5_mu, ma5_std = df_reg["sent_ma5"].mean(), df_reg["sent_ma5"].std()

        # 统一计算残差分位数
        quantiles = {}
        pred_col_map = {"Ridge": "lr_pred", "RF": "rf_pred", "XGB": "xgb_pred"}
        for name in model_info.keys():
            resids = df_reg["target_vol_t1"] - df_reg[pred_col_map[name]]
            if len(resids) < 50:
                print(f"警告：{regime} | {name} | 残差样本量={len(resids)}，分位数存在局限性")
            quantiles[name] = {
                "q_low": np.percentile(resids, 2.5),
                "q_high": np.percentile(resids, 97.5)
            }

        # 单变量扰动区间（±0.5倍标准差）
        sent_lag1_x = lag1_mu + OFFSET_RANGE * lag1_std
        sent_ma5_x = ma5_mu + OFFSET_RANGE * ma5_std
        # 双变量X轴：偏移标准差倍数
        joint_shock_label = OFFSET_RANGE  

        # 初始化绘图&结果存储
        fig, axes = plt.subplots(3, 3, figsize=(28, 18))
        fig.suptitle(f"[{regime}] Single & Joint Sensitivity | 95% Empirical Band (Volatility)", fontsize=18)
        res = {"sent_lag1": {}, "sent_ma5": {}, "dual_sent": {}}

        # ===================== 模块1：单变量 - Sent_Lag1 =====================
        for name, info in model_info.items():
            vol_mean = []
            q_low, q_high = quantiles[name]["q_low"], quantiles[name]["q_high"]

            for x in sent_lag1_x:
                x_clip = np.clip(x, sent_lag1_lower, sent_lag1_upper)
                X_in = pd.DataFrame([[x_clip, S_fix, r_fix, rv_fix, ma5_mu]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                if info["linear"]:
                    vol = info["model"].predict(scaler.transform(X_in))[0]
                else:
                    vol = info["model"].predict(X_in)[0]
                vol_mean.append(vol)

            # 波动率非负批量约束
            vol_arr = np.array(vol_mean)
            vol_lb = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub = vol_arr + q_high

            # 期权价格上下界（修复：过滤NaN，避免绘图失败）
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()

            # 存储结果
            res["sent_lag1"][name] = {"vol": vol_arr}

            # 绘图
            axes[0,0].plot(sent_lag1_x, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,0].fill_between(sent_lag1_x, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,0].plot(sent_lag1_x, price_mean, color=info["color"], linewidth=2.5)
            axes[1,0].fill_between(sent_lag1_x, price_lb, price_ub, color=info["color"], alpha=0.15)
            
            # 弹性计算
            dvol = np.gradient(vol_arr, sent_lag1_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_lag1_x - lag1_mu) / lag1_std > X_THRESHOLD)
                elas = np.where(mask, dvol * (sent_lag1_x / vol_arr), np.nan)
            axes[2,0].plot(sent_lag1_x, elas, color=info["color"], linewidth=2)

        # 图表样式
        axes[0,0].set_title("Volatility | Sent_Lag1 (±0.5σ)"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        axes[1,0].set_title("Option Price | Sent_Lag1"); axes[1,0].grid(alpha=0.3)
        axes[2,0].set_title("Elasticity | Sent_Lag1"); axes[2,0].axhline(0, c='k', ls='--'); axes[2,0].grid(alpha=0.3)

        # ===================== 模块2：单变量 - Sent_Ma5 =====================
        for name, info in model_info.items():
            vol_mean = []
            q_low, q_high = quantiles[name]["q_low"], quantiles[name]["q_high"]

            for x in sent_ma5_x:
                x_clip = np.clip(x, sent_ma5_lower, sent_ma5_upper)
                X_in = pd.DataFrame([[lag1_mu, S_fix, r_fix, rv_fix, x_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                if info["linear"]:
                    vol = info["model"].predict(scaler.transform(X_in))[0]
                else:
                    vol = info["model"].predict(X_in)[0]
                vol_mean.append(vol)

            vol_arr = np.array(vol_mean)
            vol_lb = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub = vol_arr + q_high

            # 期权价格上下界
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()

            res["sent_ma5"][name] = {"vol": vol_arr}

            axes[0,1].plot(sent_ma5_x, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,1].fill_between(sent_ma5_x, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,1].plot(sent_ma5_x, price_mean, color=info["color"], linewidth=2.5)
            axes[1,1].fill_between(sent_ma5_x, price_lb, price_ub, color=info["color"], alpha=0.15)
            
            # 弹性计算
            dvol = np.gradient(vol_arr, sent_ma5_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_ma5_x - ma5_mu) / ma5_std > X_THRESHOLD)
                elas = np.where(mask, dvol * (sent_ma5_x / vol_arr), np.nan)
            axes[2,1].plot(sent_ma5_x, elas, color=info["color"], linewidth=2)

        axes[0,1].set_title("Volatility | Sent_Ma5 (±0.5σ)"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
        axes[1,1].set_title("Option Price | Sent_Ma5"); axes[1,1].grid(alpha=0.3)
        axes[2,1].set_title("Elasticity | Sent_Ma5"); axes[2,1].axhline(0, c='k', ls='--'); axes[2,1].grid(alpha=0.3)

        # ===================== 模块3：双变量联合扰动 =====================
        for name, info in model_info.items():
            vol_mean = []
            q_low, q_high = quantiles[name]["q_low"], quantiles[name]["q_high"]

            for offset in OFFSET_RANGE:
                lag_val = lag1_mu + offset * lag1_std
                ma5_val = ma5_mu + offset * ma5_std
                # 防OOD裁剪
                lag_clip = np.clip(lag_val, sent_lag1_lower, sent_lag1_upper)
                ma5_clip = np.clip(ma5_val, sent_ma5_lower, sent_ma5_upper)

                X_in = pd.DataFrame([[lag_clip, S_fix, r_fix, rv_fix, ma5_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                if info["linear"]:
                    vol = info["model"].predict(scaler.transform(X_in))[0]
                else:
                    vol = info["model"].predict(X_in)[0]
                vol_mean.append(vol)

            vol_arr = np.array(vol_mean)
            vol_lb = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub = vol_arr + q_high

            # 期权价格上下界
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()

            res["dual_sent"][name] = {"vol": vol_arr}

            axes[0,2].plot(joint_shock_label, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,2].fill_between(joint_shock_label, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,2].plot(joint_shock_label, price_mean, color=info["color"], linewidth=2.5)
            axes[1,2].fill_between(joint_shock_label, price_lb, price_ub, color=info["color"], alpha=0.15)
            
            # 弹性计算
            dvol = np.gradient(vol_arr, joint_shock_label)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(joint_shock_label) > X_THRESHOLD)
                elas = np.where(mask, dvol * (joint_shock_label / vol_arr), np.nan)
            axes[2,2].plot(joint_shock_label, elas, color=info["color"], linewidth=2)

        axes[0,2].set_title("Volatility | Joint Shock (×σ)"); axes[0,2].legend(); axes[0,2].grid(alpha=0.3)
        axes[1,2].set_title("Option Price | Joint Shock"); axes[1,2].grid(alpha=0.3)
        axes[2,2].set_title("Elasticity | Joint Shock"); axes[2,2].set_xlabel("Joint Sentiment Shock (×Std)")
        axes[2,2].axhline(0, c='k', ls='--'); axes[2,2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"plots/sensitivity_final_{regime}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ===================== 弹性结果输出 =====================
        print("\n" + "="*100)
        print(f"[{regime}] 平均弹性结果（统一阈值：≥0.05σ）")
        print("="*100)
        for name in model_info.keys():
            # 单变量Lag1
            vol1 = res["sent_lag1"][name]["vol"]
            dvol1 = np.gradient(vol1, sent_lag1_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask1 = (np.abs(vol1) > 1e-6) & (np.abs(sent_lag1_x - lag1_mu) / lag1_std > X_THRESHOLD)
                elas1 = np.nanmean(np.where(mask1, dvol1 * (sent_lag1_x / vol1), np.nan))
            
            # 单变量Ma5
            vol2 = res["sent_ma5"][name]["vol"]
            dvol2 = np.gradient(vol2, sent_ma5_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask2 = (np.abs(vol2) > 1e-6) & (np.abs(sent_ma5_x - ma5_mu) / ma5_std > X_THRESHOLD)
                elas2 = np.nanmean(np.where(mask2, dvol2 * (sent_ma5_x / vol2), np.nan))
            
            # 双变量
            vol3 = res["dual_sent"][name]["vol"]
            dvol3 = np.gradient(vol3, joint_shock_label)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask3 = (np.abs(vol3) > 1e-6) & (np.abs(joint_shock_label) > X_THRESHOLD)
                elas3 = np.nanmean(np.where(mask3, dvol3 * (joint_shock_label / vol3), np.nan))
            
            print(f"{name:6s} | Lag1={elas1:.4f} | Ma5={elas2:.4f} | Joint={elas3:.4f}")

plot_final_uncertainty_sensitivity()

# ===================== 希腊字母 =====================
def calculate_greeks(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    invalid = (sigma <= 1e-6) | (T <= 0) | np.isnan(S) | np.isnan(r) | np.isnan(sigma)
    sigma_safe = np.where(invalid, 1.0, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * np.sqrt(T))
    d2 = d1 - sigma_safe * np.sqrt(T)
    delta = np.where(invalid, np.nan, norm.cdf(d1))
    gamma = np.where(invalid, np.nan, norm.pdf(d1) / (S * sigma_safe * np.sqrt(T)))
    vega = np.where(invalid, np.nan, S * norm.pdf(d1) * np.sqrt(T))
    theta = np.where(invalid, np.nan, - (S * norm.pdf(d1) * sigma_safe) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2))
    rho = np.where(invalid, np.nan, K * T * np.exp(-r*T) * norm.cdf(d2))
    return delta, gamma, vega, theta, rho

df_test['delta'], df_test['gamma'], df_test['vega'], df_test['theta'], df_test['rho'] = calculate_greeks(
    df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol']
)
greeks_df = df_test[['date', 'delta', 'gamma', 'vega', 'theta', 'rho', 'pred_vol', 'two_step_price']].copy()
greeks_df.to_csv("reports/option_greeks.csv", index=False)

# ===================== 最终波动率预测总表 =====================
vol_summary = []
model_vol_map = [
    ["Linear Regression (Ridge L2)", lr_pred],
    ["Random Forest", rf_pred],
    ["XGBoost", xgb_pred]
]
for name, pred in model_vol_map:
    _, _, rmse, r2, dir_acc = evaluate(y_test, pred)
    cv_rmse_list, _ = expanding_window_validation(
        X_train, y_train, 
        lr_model if name=="Linear Regression (Ridge L2)" else best_rf if name=="Random Forest" else best_xgb, 
        tscv, is_linear=(name=="Linear Regression (Ridge L2)"), scaler=scaler
    )
    cv_rmse = np.mean(cv_rmse_list)
    vol_summary.append([name, round(rmse,4), round(r2,4), round(dir_acc,4), round(cv_rmse,4)])

# BSM 波动率基准
_, _, bsm_rmse, bsm_r2, bsm_dir = evaluate(y_test, df_test["rv_lag1"])
vol_summary.append(["BSM Baseline (Lagged Vol)", round(bsm_rmse,4), round(bsm_r2,4), round(bsm_dir,4), "N/A"])

vol_summary_df = pd.DataFrame(vol_summary, columns=[
    "Model", "Volatility_RMSE", "Volatility_R2", "Directional_Acc", "Expanding_Window_CV_RMSE"
])
vol_summary_df.to_csv("reports/volatility_prediction_summary.csv", index=False)

print("\n" + "="*70)
print("波动率预测模型性能对比")
print("="*70)
print(vol_summary_df)
print("表格已保存：reports/volatility_prediction_summary.csv")

# ===================== 最终输出 =====================
print("\n" + "="*50)
print(f"Best Model: {best_model_name}")
print(f"Volatility RMSE: {results_df.loc[best_idx, 'RMSE']:.4f}")
print(f"Directional Accuracy: {results_df.loc[best_idx, 'Directional_Acc']:.2%}")
print(f"Expanding Window CV RMSE: {np.mean(expanding_rmse_list):.4f}")
print("="*50)

# 显著性检验
from scipy.stats import mannwhitneyu
model_config = {"Linear Regression (Ridge)": "lr_pred", "Random Forest": "rf_pred", "XGBoost": "xgb_pred"}
low_mask = df_test["regime"] == "Low_Vol"
high_mask = df_test["regime"] == "High_Vol"

print("\n" + "="*70)
print("三个模型 - 高低波动误差差异显著性检验 (Mann-Whitney U)")
print("="*70)
significant_models = []
for model_name, pred_col in model_config.items():
    err_low = np.abs(y_test[low_mask] - df_test.loc[low_mask, pred_col])
    err_high = np.abs(y_test[high_mask] - df_test.loc[high_mask, pred_col])
    stat, p_val = mannwhitneyu(err_low, err_high, alternative="two-sided")
    is_significant = p_val < 0.05
    if is_significant:
        significant_models.append(model_name)
    print(f"【{model_name}】p值 = {p_val:.4f} | 差异：{'统计显著' if is_significant else '不显著'}")

print(f"\n检验总结：{', '.join(significant_models) if significant_models else '无'} 模型呈现显著差异")
print("\n全部任务完成！")