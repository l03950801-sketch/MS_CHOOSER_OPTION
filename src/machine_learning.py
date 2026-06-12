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
from scipy.integrate import quad          # ← 新增：Heston 数值积分需要
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

    lr_params = {"alpha": [0.01, 0.1, 1, 10, 100], "random_state": [SEED]}
    lr_search = GridSearchCV(Ridge(), lr_params, cv=tscv, scoring="neg_root_mean_squared_error")
    lr_search.fit(X_train_scaled, y_train)
    lr_model = lr_search.best_estimator_

    rf_params = {"n_estimators": [100,200], "max_depth": [2,3,5], "min_samples_split": [2,4], "min_samples_leaf": [1,3], "random_state": [SEED]}
    rf_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_root_mean_squared_error")
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

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

baseline_map = {"BSM Baseline (Lagged Vol)": "rv_lag1"}
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

# ===================== Parity Plot =====================
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

def plot_standardized_coefficients(model, features):
    coef = model.coef_
    coef_df = pd.DataFrame({"Feature": features, "Standardized_Coefficient": coef})
    coef_df["Abs_Coeff"] = coef_df["Standardized_Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coeff", ascending=False).reset_index(drop=True)
    coef_df.to_csv("reports/standardized_coefficients.csv", index=False)
    print("\n" + "="*65)
    print("Linear Regression (Ridge) 标准化回归系数")
    print("="*65)
    print(coef_df[["Feature", "Standardized_Coefficient"]].round(4))
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
        if len(sv.shape) == 1:
            sv = sv.reshape(-1, 1)
    feature_pairs = list(itertools.combinations(FEATURES, 2))
    for feat1, feat2 in feature_pairs:
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feat1, sv, X_test, interaction_index=feat2, show=False, alpha=0.6)
            plt.title(f"{model_name} | {feat1} × {feat2} Feature Interaction", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"plots/shap_interact_{model_name}_{feat1}_{feat2}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[{model_name}] 生成 {feat1}-{feat2} 交互图失败: {str(e)}")
            plt.close()

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


# ============================================================
# 新增模块：Heston + SABR 定价函数
# ============================================================

def heston_char_func(phi, S, K, T, r, v0, kappa, theta, xi, rho):
    i = complex(0, 1)
    x = np.log(S / K)
    a = kappa * theta
    d = np.sqrt((rho * xi * i * phi - kappa)**2 + xi**2 * (i * phi + phi**2))
    g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)
    exp_dT = np.exp(-d * T)
    C = (r * i * phi * T
         + (a / xi**2) * ((kappa - rho * xi * i * phi - d) * T
                          - 2 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((kappa - rho * xi * i * phi - d) / xi**2
         * (1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * v0 + i * phi * x)

def heston_integrand(phi, S, K, T, r, v0, kappa, theta, xi, rho, j):
    i = complex(0, 1)
    if j == 1:
        cf  = heston_char_func(phi - i, S, K, T, r, v0, kappa, theta, xi, rho)
        cf0 = heston_char_func(-i,      S, K, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-i * phi * np.log(K)) * cf / (i * phi * cf0))
    else:
        cf = heston_char_func(phi, S, K, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-i * phi * np.log(K)) * cf / (i * phi))

def heston_price_single(S, K, T, r, v0, kappa, theta, xi, rho):
    if T <= 0 or S <= 0 or K <= 0 or v0 <= 0:
        return np.nan
    try:
        P1, _ = quad(heston_integrand, 1e-6, 50,
                     args=(S, K, T, r, v0, kappa, theta, xi, rho, 1),
                     limit=100, epsabs=1e-6)
        P2, _ = quad(heston_integrand, 1e-6, 50,
                     args=(S, K, T, r, v0, kappa, theta, xi, rho, 2),
                     limit=100, epsabs=1e-6)
        P1 = 0.5 + P1 / np.pi
        P2 = 0.5 + P2 / np.pi
        price = S * P1 - K * np.exp(-r * T) * P2
        return max(price, max(S - K * np.exp(-r * T), 0))
    except Exception:
        return np.nan

def heston_price_vec(S_arr, K, T, r_arr, pred_vol_arr,
                     kappa=2.0, theta=0.04, xi=0.5, rho=-0.7):
    prices = []
    for S, r, sigma in zip(S_arr, r_arr, pred_vol_arr):
        v0 = max(sigma**2, 1e-6)
        prices.append(heston_price_single(S, K, T, r, v0, kappa, theta, xi, rho))
    return np.array(prices)

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    if T <= 0 or F <= 0 or K <= 0 or alpha <= 0:
        return np.nan
    if abs(F - K) < 1e-8:
        FK_mid = F ** (1 - beta)
        term1 = alpha / FK_mid
        term2 = (1 + ((1 - beta)**2 / 24 * alpha**2 / FK_mid**2
                      + rho * beta * nu * alpha / (4 * FK_mid)
                      + (2 - 3 * rho**2) / 24 * nu**2) * T)
        return term1 * term2
    log_FK  = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z       = (nu / alpha) * FK_beta * log_FK
    x_z     = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    z_over_xz = 1.0 if abs(x_z) < 1e-10 else z / x_z
    numerator  = alpha / (FK_beta * (1 + (1 - beta)**2 / 24 * log_FK**2
                                      + (1 - beta)**4 / 1920 * log_FK**4))
    correction = (1 + ((1 - beta)**2 / 24 * alpha**2 / FK_beta**2
                       + rho * beta * nu * alpha / (4 * FK_beta)
                       + (2 - 3 * rho**2) / 24 * nu**2) * T)
    return numerator * z_over_xz * correction

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 1e-6 or T <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def sabr_price_vec(S_arr, K, T, r_arr, pred_vol_arr,
                   beta=0.5, rho=-0.3, nu=0.4):
    prices = []
    for S, r, alpha in zip(S_arr, r_arr, pred_vol_arr):
        F     = S * np.exp(r * T)
        alpha_safe = max(alpha, 1e-4)
        sigma_sabr = sabr_implied_vol(F, K, T, alpha_safe, beta, rho, nu)
        if sigma_sabr is None or np.isnan(sigma_sabr) or sigma_sabr <= 0:
            prices.append(np.nan)
        else:
            prices.append(black_scholes_call(S, K, T, r, sigma_sabr))
    return np.array(prices)

def ensemble_price(bsm_prices, heston_prices, sabr_prices, regime_series,
                   weights_low=(0.5, 0.3, 0.2),
                   weights_high=(0.2, 0.3, 0.5)):
    ensemble = np.full(len(bsm_prices), np.nan)
    for i, regime in enumerate(regime_series):
        w = weights_low if regime == "Low_Vol" else weights_high
        prices  = np.array([bsm_prices[i], heston_prices[i], sabr_prices[i]])
        weights = np.array(w)
        valid   = np.isfinite(prices)
        if valid.sum() == 0:
            continue
        wv = weights[valid] / weights[valid].sum()
        ensemble[i] = np.dot(prices[valid], wv)
    return ensemble


# ============================================================
# 新增模块：Validation-set Grid Search 参数校准
# ─ 在 train set 后 30% 作为 validation（时序上早于 test）
# ─ 穷举参数组合，选出 validation RMSE 最小的参数
# ─ 满足 Heston Feller 条件：2κθ > ξ²
# ============================================================

def calibrate_heston(df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate):
    """
    在 validation set 上对 Heston 做 grid search。

    搜索空间（文献约束范围）：
      kappa : [0.5, 1.0, 2.0, 4.0]        均值回归速度，> 0
      theta : [0.02, 0.04, 0.06]           长期方差，> 0
      xi    : [0.2, 0.5, 0.8]             vol of vol，> 0
      rho   : [-0.9, -0.7, -0.5, -0.3]    相关系数，(-1, 1)
    硬约束：Feller 条件 2κθ > ξ²（确保方差过程不触零）
    """
    S_val   = df_val['S'].values
    r_val   = df_val['r'].values
    vol_val = df_val['rv_lag1'].values
    y_val   = df_val['target_vol_t1'].values
    true_p  = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val)

    param_grid = {
        'kappa': [0.5, 1.0, 2.0, 4.0],
        'theta': [0.02, 0.04, 0.06],
        'xi':    [0.2, 0.5, 0.8],
        'rho':   [-0.9, -0.7, -0.5, -0.3],
    }

    best_rmse   = np.inf
    best_params = {'kappa': 2.0, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7}
    records     = []

    total = (len(param_grid['kappa']) * len(param_grid['theta'])
             * len(param_grid['xi'])  * len(param_grid['rho']))
    print(f"\nHeston grid search: {total} 组合...")

    for kappa in param_grid['kappa']:
        for theta in param_grid['theta']:
            for xi in param_grid['xi']:
                # ── Feller 条件检查 ──────────────────────────
                # 2κθ > ξ²  →  方差过程永远为正
                if 2 * kappa * theta <= xi ** 2:
                    continue
                for rho in param_grid['rho']:
                    preds = heston_price_vec(
                        S_val, STRIKE, T_MATURITY, r_val, vol_val,
                        kappa=kappa, theta=theta, xi=xi, rho=rho
                    )
                    valid = np.isfinite(preds) & np.isfinite(true_p)
                    if valid.sum() < 10:
                        continue
                    _, _, rmse, r2, _ = evaluate(true_p[valid], preds[valid])
                    records.append({
                        'kappa': kappa, 'theta': theta,
                        'xi': xi, 'rho': rho,
                        'feller': round(2*kappa*theta - xi**2, 4),
                        'val_RMSE': round(rmse, 4),
                        'val_R2':   round(r2, 4),
                    })
                    if rmse < best_rmse:
                        best_rmse   = rmse
                        best_params = {'kappa': kappa, 'theta': theta,
                                       'xi': xi, 'rho': rho}

    calib_df = pd.DataFrame(records).sort_values('val_RMSE')
    calib_df.to_csv("reports/heston_calibration_grid.csv", index=False)

    print(f"Heston 最优参数：{best_params}  |  Val RMSE={best_rmse:.4f}")
    print(f"Feller 条件验证：2κθ - ξ² = "
          f"{2*best_params['kappa']*best_params['theta'] - best_params['xi']**2:.4f} > 0 ✓")
    return best_params, calib_df


def calibrate_sabr(df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate):
    """
    在 validation set 上对 SABR 做 grid search。

    搜索空间：
      beta : [0.0, 0.5, 1.0]     弹性参数（股票常用 0.5）
      rho  : [-0.5, -0.3, 0.0]   相关系数
      nu   : [0.2, 0.4, 0.6, 0.8] vol of vol

    alpha（初始波动率）直接用每行的 rv_lag1，不在 grid 里搜索，
    因为你的 alpha 本质上就是 ML 预测的波动率。
    """
    S_val   = df_val['S'].values
    r_val   = df_val['r'].values
    vol_val = df_val['rv_lag1'].values
    y_val   = df_val['target_vol_t1'].values
    true_p  = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val)

    param_grid = {
        'beta': [0.0, 0.5, 1.0],
        'rho':  [-0.5, -0.3, 0.0],
        'nu':   [0.2, 0.4, 0.6, 0.8],
    }

    best_rmse   = np.inf
    best_params = {'beta': 0.5, 'rho': -0.3, 'nu': 0.4}
    records     = []

    total = len(param_grid['beta']) * len(param_grid['rho']) * len(param_grid['nu'])
    print(f"\nSABR grid search: {total} 组合...")

    for beta in param_grid['beta']:
        for rho in param_grid['rho']:
            for nu in param_grid['nu']:
                preds = sabr_price_vec(
                    S_val, STRIKE, T_MATURITY, r_val, vol_val,
                    beta=beta, rho=rho, nu=nu
                )
                valid = np.isfinite(preds) & np.isfinite(true_p)
                if valid.sum() < 10:
                    continue
                _, _, rmse, r2, _ = evaluate(true_p[valid], preds[valid])
                records.append({
                    'beta': beta, 'rho': rho, 'nu': nu,
                    'val_RMSE': round(rmse, 4),
                    'val_R2':   round(r2, 4),
                })
                if rmse < best_rmse:
                    best_rmse   = rmse
                    best_params = {'beta': beta, 'rho': rho, 'nu': nu}

    calib_df = pd.DataFrame(records).sort_values('val_RMSE')
    calib_df.to_csv("reports/sabr_calibration_grid.csv", index=False)

    print(f"SABR 最优参数：{best_params}  |  Val RMSE={best_rmse:.4f}")
    return best_params, calib_df


# ============================================================
# 新增模块：三模型定价主流水线
# ============================================================

def run_three_model_pricing(df_test, STRIKE, T_MATURITY, y_pred_vol, y_test,
                            black_scholes_vec, evaluate,
                            heston_params, sabr_params,
                            ensemble_weights_low=(0.5, 0.3, 0.2),
                            ensemble_weights_high=(0.2, 0.3, 0.5)):
    """
    三模型定价 + Ensemble，使用校准后的参数。
    heston_params / sabr_params 由 calibrate_* 函数返回，不再硬编码。
    """
    S_arr      = df_test['S'].values
    r_arr      = df_test['r'].values
    regime_arr = df_test['regime'].values

    print("\n" + "="*60)
    print("三模型定价计算中（使用校准参数）...")
    print(f"  Heston: {heston_params}")
    print(f"  SABR:   {sabr_params}")
    print("="*60)

    bsm_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol)

    print("计算 Heston 价格（逐行积分，需要约10-30秒）...")
    heston_prices = heston_price_vec(
        S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **heston_params
    )

    print("计算 SABR 价格...")
    sabr_prices = sabr_price_vec(
        S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **sabr_params
    )

    ensemble_prices = ensemble_price(
        bsm_prices, heston_prices, sabr_prices, regime_arr,
        weights_low=ensemble_weights_low,
        weights_high=ensemble_weights_high
    )

    true_bsm_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr,
                                         np.asarray(y_test))

    print("\n" + "="*60)
    print("三模型定价性能对比（benchmark = true BSM price）")
    print("="*60)

    pricing_results = []
    for name, pred_p in [("BSM", bsm_prices), ("Heston", heston_prices),
                          ("SABR", sabr_prices), ("Ensemble", ensemble_prices)]:
        valid = np.isfinite(pred_p) & np.isfinite(true_bsm_prices)
        if valid.sum() == 0:
            print(f"{name}: 无有效预测")
            continue
        mse, mae, rmse, r2, _ = evaluate(true_bsm_prices[valid], pred_p[valid])
        pricing_results.append([name, round(rmse,4), round(mae,4), round(r2,4)])
        print(f"{name:10s} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    pd.DataFrame(pricing_results, columns=["Model","RMSE","MAE","R2"]).to_csv(
        "reports/three_model_pricing_comparison.csv", index=False)

    print("\n" + "="*60)
    print("分 Regime 定价性能对比")
    print("="*60)

    regime_pricing_results = []
    for regime in ["Low_Vol", "High_Vol"]:
        mask     = regime_arr == regime
        true_sub = true_bsm_prices[mask]
        for name, pred_p in [("BSM", bsm_prices), ("Heston", heston_prices),
                              ("SABR", sabr_prices), ("Ensemble", ensemble_prices)]:
            pred_sub = pred_p[mask]
            valid    = np.isfinite(pred_sub) & np.isfinite(true_sub)
            if valid.sum() == 0:
                continue
            _, _, rmse, r2, _ = evaluate(true_sub[valid], pred_sub[valid])
            regime_pricing_results.append([regime, name, mask.sum(), round(rmse,4), round(r2,4)])
            print(f"{regime:10s} | {name:10s} | N={mask.sum()} | RMSE={rmse:.4f} | R²={r2:.4f}")

    pd.DataFrame(regime_pricing_results,
                 columns=["Regime","Model","N","RMSE","R2"]).to_csv(
        "reports/regime_three_model_pricing.csv", index=False)

    output_df = df_test[['date','regime']].copy()
    output_df['pred_vol']       = y_pred_vol
    output_df['true_bsm_price'] = true_bsm_prices
    output_df['bsm_price']      = bsm_prices
    output_df['heston_price']   = heston_prices
    output_df['sabr_price']     = sabr_prices
    output_df['ensemble_price'] = ensemble_prices
    output_df.to_csv("reports/three_model_pricing_full.csv", index=False)

    # 时间序列图
    plt.figure(figsize=(14, 6))
    dates = df_test['date'].values
    plt.plot(dates, true_bsm_prices, label='True BSM Price',  color='black',   linewidth=1.5, alpha=0.8)
    plt.plot(dates, bsm_prices,      label='BSM (pred vol)',  color='#378ADD', linewidth=1.2, linestyle='--')
    plt.plot(dates, heston_prices,   label='Heston',          color='#1D9E75', linewidth=1.2, linestyle='-.')
    plt.plot(dates, sabr_prices,     label='SABR',            color='#BA7517', linewidth=1.2, linestyle=':')
    plt.plot(dates, ensemble_prices, label='Ensemble',        color='#D4537E', linewidth=2.0)
    plt.xlabel('Date'); plt.ylabel('Option Price')
    plt.title('Three-Model Option Pricing Comparison')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("plots/three_model_pricing_timeseries.png", dpi=300)
    plt.close()

    # Regime RMSE 条形图
    if regime_pricing_results:
        rpr_df = pd.DataFrame(regime_pricing_results, columns=["Regime","Model","N","RMSE","R2"])
        plt.figure(figsize=(10, 5))
        sns.barplot(data=rpr_df, x="Regime", y="RMSE", hue="Model",
                    palette=["#378ADD","#1D9E75","#BA7517","#D4537E"])
        plt.title("Three-Model Pricing RMSE by Volatility Regime")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("plots/three_model_regime_rmse.png", dpi=300)
        plt.close()

    print("\n三模型定价图表已保存至 plots/ 目录")
    return output_df


# ============================================================
# 新增模块：OOS 权重推导 + 敏感性分析
# ============================================================

def derive_oos_weights(df_val, STRIKE, T_MATURITY,
                       black_scholes_vec, evaluate,
                       heston_params, sabr_params):
    S_val   = df_val['S'].values
    r_val   = df_val['r'].values
    vol_val = df_val['rv_lag1'].values
    y_val   = df_val['target_vol_t1'].values
    true_p  = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val)

    bsm_val    = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, vol_val)
    heston_val = heston_price_vec(S_val, STRIKE, T_MATURITY, r_val, vol_val, **heston_params)
    sabr_val   = sabr_price_vec(S_val, STRIKE, T_MATURITY, r_val, vol_val, **sabr_params)

    rmse_results = {}
    for name, preds in [('BSM', bsm_val), ('Heston', heston_val), ('SABR', sabr_val)]:
        valid = np.isfinite(preds) & np.isfinite(true_p)
        if valid.sum() == 0:
            rmse_results[name] = np.inf
            continue
        _, _, rmse, _, _ = evaluate(true_p[valid], preds[valid])
        rmse_results[name] = rmse

    print("\n" + "="*50)
    print("Validation Set RMSE（用于推导 OOS 权重）")
    print("="*50)
    for name, rmse in rmse_results.items():
        print(f"  {name:8s}: RMSE = {rmse:.4f}")

    inv_rmse   = {k: 1.0/v if v > 0 else 0 for k, v in rmse_results.items()}
    total      = sum(inv_rmse.values())
    oos_w      = {k: v/total for k, v in inv_rmse.items()}

    print(f"\nOOS 权重：BSM={oos_w['BSM']:.3f} | Heston={oos_w['Heston']:.3f} | SABR={oos_w['SABR']:.3f}")
    w_tuple = (oos_w['BSM'], oos_w['Heston'], oos_w['SABR'])
    return w_tuple, rmse_results


def heston_sensitivity(df_test, STRIKE, T_MATURITY, y_pred_vol,
                       black_scholes_vec, evaluate, best_heston_params):
    """
    单变量扫描：每次只改动一个参数，其余固定为校准最优值。
    基准值 = 校准得到的 best_heston_params，不再是主观设定。
    """
    S_arr = df_test['S'].values
    r_arr = df_test['r'].values
    y_true_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr,
                                       np.asarray(df_test['target_vol_t1']))
    param_grid = {
        'kappa': [0.5, 1.0, 2.0, 4.0],
        'rho':   [-0.9, -0.7, -0.5, -0.3],
        'xi':    [0.2, 0.5, 0.8],
        'theta': [0.02, 0.04, 0.06],
    }
    base = best_heston_params.copy()   # ← 使用校准值，非主观值
    records = []

    for param_name, values in param_grid.items():
        for val in values:
            params = base.copy()
            params[param_name] = val
            # Feller 条件保护
            if 2 * params['kappa'] * params['theta'] <= params['xi'] ** 2:
                continue
            preds = heston_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol,
                                      **params)
            valid = np.isfinite(preds) & np.isfinite(y_true_prices)
            if valid.sum() == 0:
                continue
            _, _, rmse, r2, _ = evaluate(y_true_prices[valid], preds[valid])
            records.append({'param': param_name, 'value': val,
                             'RMSE': round(rmse,4), 'R2': round(r2,4),
                             'is_best': (val == base[param_name])})

    sens_df = pd.DataFrame(records)
    sens_df.to_csv("reports/heston_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Heston Parameter Sensitivity — RMSE\n(red dot = calibrated best value)", fontsize=13)
    for ax, pname in zip(axes, param_grid.keys()):
        sub = sens_df[sens_df['param'] == pname].sort_values('value')
        ax.plot(sub['value'], sub['RMSE'], marker='o', color='#185FA5', linewidth=2)
        best_row = sub[sub['is_best']]
        if len(best_row):
            ax.scatter(best_row['value'], best_row['RMSE'],
                       color='#E24B4A', zorder=5, s=80, label='calibrated')
            ax.legend(fontsize=9)
        ax.set_xlabel(pname); ax.set_ylabel('RMSE'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/heston_sensitivity.png", dpi=300)
    plt.close()
    print("Heston 敏感性分析完成 → plots/heston_sensitivity.png")
    return sens_df


def sabr_sensitivity(df_test, STRIKE, T_MATURITY, y_pred_vol,
                     black_scholes_vec, evaluate, best_sabr_params):
    S_arr = df_test['S'].values
    r_arr = df_test['r'].values
    y_true_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr,
                                       np.asarray(df_test['target_vol_t1']))
    param_grid = {
        'beta': [0.0, 0.5, 1.0],
        'rho':  [-0.5, -0.3, 0.0],
        'nu':   [0.2, 0.4, 0.6, 0.8],
    }
    base    = best_sabr_params.copy()   # ← 使用校准值
    records = []

    for param_name, values in param_grid.items():
        for val in values:
            params = base.copy()
            params[param_name] = val
            preds = sabr_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **params)
            valid = np.isfinite(preds) & np.isfinite(y_true_prices)
            if valid.sum() == 0:
                continue
            _, _, rmse, r2, _ = evaluate(y_true_prices[valid], preds[valid])
            records.append({'param': param_name, 'value': val,
                             'RMSE': round(rmse,4), 'R2': round(r2,4),
                             'is_best': (val == base[param_name])})

    sens_df = pd.DataFrame(records)
    sens_df.to_csv("reports/sabr_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("SABR Parameter Sensitivity — RMSE\n(red dot = calibrated best value)", fontsize=13)
    for ax, pname in zip(axes, param_grid.keys()):
        sub = sens_df[sens_df['param'] == pname].sort_values('value')
        ax.plot(sub['value'], sub['RMSE'], marker='o', color='#1D9E75', linewidth=2)
        best_row = sub[sub['is_best']]
        if len(best_row):
            ax.scatter(best_row['value'], best_row['RMSE'],
                       color='#E24B4A', zorder=5, s=80, label='calibrated')
            ax.legend(fontsize=9)
        ax.set_xlabel(pname); ax.set_ylabel('RMSE'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/sabr_sensitivity.png", dpi=300)
    plt.close()
    print("SABR 敏感性分析完成 → plots/sabr_sensitivity.png")
    return sens_df


def ensemble_weight_sensitivity(df_test, STRIKE, T_MATURITY, y_pred_vol,
                                 black_scholes_vec, evaluate,
                                 heston_params, sabr_params,
                                 oos_weights=None):
    S_arr      = df_test['S'].values
    r_arr      = df_test['r'].values
    regime_arr = df_test['regime'].values
    y_true_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr,
                                       np.asarray(df_test['target_vol_t1']))

    bsm_p    = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol)
    heston_p = heston_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **heston_params)
    sabr_p   = sabr_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **sabr_params)

    weight_schemes = {
        'Equal (1/3 each)':      {'low': (1/3,1/3,1/3),   'high': (1/3,1/3,1/3)},
        'Heuristic (BSM-heavy)': {'low': (0.5,0.3,0.2),   'high': (0.2,0.3,0.5)},
        'Pure BSM':              {'low': (1.0,0.0,0.0),   'high': (1.0,0.0,0.0)},
        'Pure Heston':           {'low': (0.0,1.0,0.0),   'high': (0.0,1.0,0.0)},
        'Pure SABR':             {'low': (0.0,0.0,1.0),   'high': (0.0,0.0,1.0)},
    }
    if oos_weights is not None:
        weight_schemes['OOS-derived'] = {'low': oos_weights, 'high': oos_weights}

    records = []
    for scheme_name, w in weight_schemes.items():
        ens   = ensemble_price(bsm_p, heston_p, sabr_p, regime_arr,
                               weights_low=w['low'], weights_high=w['high'])
        valid = np.isfinite(ens) & np.isfinite(y_true_prices)
        if valid.sum() == 0:
            continue
        _, _, rmse, r2, _ = evaluate(y_true_prices[valid], ens[valid])
        records.append({'Weight Scheme': scheme_name,
                        'RMSE': round(rmse,4), 'R2': round(r2,4)})

    weight_df = pd.DataFrame(records).sort_values('RMSE')
    weight_df.to_csv("reports/ensemble_weight_sensitivity.csv", index=False)

    print("\n" + "="*60)
    print("Ensemble 权重敏感性分析")
    print("="*60)
    print(weight_df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    colors = ['#E24B4A' if i == 0 else '#B5D4F4' for i in range(len(weight_df))]
    plt.barh(weight_df['Weight Scheme'], weight_df['RMSE'], color=colors)
    plt.xlabel('RMSE (lower is better)')
    plt.title('Ensemble Weight Sensitivity\n(red = best scheme)')
    plt.grid(alpha=0.3, axis='x'); plt.tight_layout()
    plt.savefig("plots/ensemble_weight_sensitivity.png", dpi=300)
    plt.close()
    return weight_df


# ============================================================
# 执行：校准 → 三模型定价 → 敏感性分析
# ============================================================

# validation set = train set 后 30%（时序上早于 test，无泄露）
val_size = int(len(df_train) * 0.3)
df_val   = df_train.iloc[-val_size:].copy()

# 1. 参数校准（grid search on validation set）
print("\n" + "="*60)
print("Step 1: 参数校准（Validation Set Grid Search）")
print("="*60)
best_heston_params, heston_calib_df = calibrate_heston(
    df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate
)
best_sabr_params, sabr_calib_df = calibrate_sabr(
    df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate
)

# 2. OOS 权重推导（用校准后参数）
print("\n" + "="*60)
print("Step 2: OOS 权重推导")
print("="*60)
oos_w, _ = derive_oos_weights(
    df_val, STRIKE, T_MATURITY,
    black_scholes_vec, evaluate,
    best_heston_params, best_sabr_params
)

# 3. 三模型定价（用校准参数 + OOS 权重）
print("\n" + "="*60)
print("Step 3: 三模型定价")
print("="*60)
pricing_output = run_three_model_pricing(
    df_test, STRIKE, T_MATURITY, y_pred_vol, y_test,
    black_scholes_vec, evaluate,
    heston_params=best_heston_params,
    sabr_params=best_sabr_params,
    ensemble_weights_low=oos_w,
    ensemble_weights_high=oos_w
)

# 4. 敏感性分析（以校准值为基准，验证稳健性）
print("\n" + "="*60)
print("Step 4: 敏感性分析")
print("="*60)
heston_sens_df = heston_sensitivity(
    df_test, STRIKE, T_MATURITY, y_pred_vol,
    black_scholes_vec, evaluate, best_heston_params
)
sabr_sens_df = sabr_sensitivity(
    df_test, STRIKE, T_MATURITY, y_pred_vol,
    black_scholes_vec, evaluate, best_sabr_params
)
weight_sens_df = ensemble_weight_sensitivity(
    df_test, STRIKE, T_MATURITY, y_pred_vol,
    black_scholes_vec, evaluate,
    best_heston_params, best_sabr_params,
    oos_weights=oos_w
)

# 保存校准参数供报告引用
calib_summary = pd.DataFrame([
    {'Model': 'Heston', 'Parameter': k, 'Calibrated_Value': v,
     'Constraint': 'Feller: 2κθ>ξ²' if k == 'xi' else ''}
    for k, v in best_heston_params.items()
] + [
    {'Model': 'SABR', 'Parameter': k, 'Calibrated_Value': v, 'Constraint': ''}
    for k, v in best_sabr_params.items()
])
calib_summary.to_csv("reports/calibrated_parameters.csv", index=False)
print("\n校准参数已保存：reports/calibrated_parameters.csv")


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

# ===================== 情绪敏感性分析 =====================
def plot_final_uncertainty_sensitivity():
    plt.rcParams['axes.unicode_minus'] = False

    sent_lag1_lower = df_train['sent_lag1'].quantile(0.01)
    sent_lag1_upper = df_train['sent_lag1'].quantile(0.99)
    sent_ma5_lower  = df_train['sent_ma5'].quantile(0.01)
    sent_ma5_upper  = df_train['sent_ma5'].quantile(0.99)

    model_info = {
        "Ridge": {"model": lr_model,  "linear": True,  "color": "#79c0f2"},
        "RF":    {"model": best_rf,   "linear": False, "color": "#5BE1A0"},
        "XGB":   {"model": best_xgb,  "linear": False, "color": "#bd6aaa"}
    }
    N_POINTS     = 100
    VOL_FLOOR    = 1e-4
    X_THRESHOLD  = 0.05
    OFFSET_RANGE = np.linspace(-0.5, 0.5, N_POINTS)

    df_low  = df_test[df_test['regime'] == 'Low_Vol'].copy()
    df_high = df_test[df_test['regime'] == 'High_Vol'].copy()

    regime_cfg = {
        "Low_Vol":  {"df": df_low,  "S": df_low['S_lag1'].median(),
                     "r": df_low['r_lag1'].median(),  "rv": df_low['rv_lag1'].median()},
        "High_Vol": {"df": df_high, "S": df_high['S_lag1'].median(),
                     "r": df_high['r_lag1'].median(), "rv": df_high['rv_lag1'].median()},
    }

    for regime, cfg in regime_cfg.items():
        df_reg = cfg["df"]
        S_fix, r_fix, rv_fix = cfg["S"], cfg["r"], cfg["rv"]
        lag1_mu, lag1_std = df_reg["sent_lag1"].mean(), df_reg["sent_lag1"].std()
        ma5_mu,  ma5_std  = df_reg["sent_ma5"].mean(),  df_reg["sent_ma5"].std()

        quantiles    = {}
        pred_col_map = {"Ridge": "lr_pred", "RF": "rf_pred", "XGB": "xgb_pred"}
        for name in model_info.keys():
            resids = df_reg["target_vol_t1"] - df_reg[pred_col_map[name]]
            if len(resids) < 50:
                print(f"警告：{regime} | {name} | 残差样本量={len(resids)}")
            quantiles[name] = {"q_low":  np.percentile(resids, 2.5),
                                "q_high": np.percentile(resids, 97.5)}

        sent_lag1_x    = lag1_mu + OFFSET_RANGE * lag1_std
        sent_ma5_x     = ma5_mu  + OFFSET_RANGE * ma5_std
        joint_shock_label = OFFSET_RANGE

        fig, axes = plt.subplots(3, 3, figsize=(28, 18))
        fig.suptitle(f"[{regime}] Single & Joint Sensitivity | 95% Empirical Band (Volatility)", fontsize=18)
        res = {"sent_lag1": {}, "sent_ma5": {}, "dual_sent": {}}

        for name, info in model_info.items():
            q_low, q_high = quantiles[name]["q_low"], quantiles[name]["q_high"]

            # ── Sent_Lag1 ──
            vol_mean = []
            for x in sent_lag1_x:
                x_clip = np.clip(x, sent_lag1_lower, sent_lag1_upper)
                X_in = pd.DataFrame([[x_clip, S_fix, r_fix, rv_fix, ma5_mu]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                vol_mean.append(info["model"].predict(scaler.transform(X_in))[0]
                                if info["linear"] else info["model"].predict(X_in)[0])
            vol_arr  = np.array(vol_mean)
            vol_lb   = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub   = vol_arr + q_high
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()
            res["sent_lag1"][name] = {"vol": vol_arr}
            axes[0,0].plot(sent_lag1_x, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,0].fill_between(sent_lag1_x, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,0].plot(sent_lag1_x, price_mean, color=info["color"], linewidth=2.5)
            axes[1,0].fill_between(sent_lag1_x, price_lb, price_ub, color=info["color"], alpha=0.15)
            dvol = np.gradient(vol_arr, sent_lag1_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_lag1_x - lag1_mu) / lag1_std > X_THRESHOLD)
                elas = np.where(mask, dvol * (sent_lag1_x / vol_arr), np.nan)
            axes[2,0].plot(sent_lag1_x, elas, color=info["color"], linewidth=2)

            # ── Sent_Ma5 ──
            vol_mean = []
            for x in sent_ma5_x:
                x_clip = np.clip(x, sent_ma5_lower, sent_ma5_upper)
                X_in = pd.DataFrame([[lag1_mu, S_fix, r_fix, rv_fix, x_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                vol_mean.append(info["model"].predict(scaler.transform(X_in))[0]
                                if info["linear"] else info["model"].predict(X_in)[0])
            vol_arr  = np.array(vol_mean)
            vol_lb   = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub   = vol_arr + q_high
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()
            res["sent_ma5"][name] = {"vol": vol_arr}
            axes[0,1].plot(sent_ma5_x, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,1].fill_between(sent_ma5_x, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,1].plot(sent_ma5_x, price_mean, color=info["color"], linewidth=2.5)
            axes[1,1].fill_between(sent_ma5_x, price_lb, price_ub, color=info["color"], alpha=0.15)
            dvol = np.gradient(vol_arr, sent_ma5_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_ma5_x - ma5_mu) / ma5_std > X_THRESHOLD)
                elas = np.where(mask, dvol * (sent_ma5_x / vol_arr), np.nan)
            axes[2,1].plot(sent_ma5_x, elas, color=info["color"], linewidth=2)

            # ── Joint Shock ──
            vol_mean = []
            for offset in OFFSET_RANGE:
                lag_clip = np.clip(lag1_mu + offset*lag1_std, sent_lag1_lower, sent_lag1_upper)
                ma5_clip = np.clip(ma5_mu  + offset*ma5_std,  sent_ma5_lower,  sent_ma5_upper)
                X_in = pd.DataFrame([[lag_clip, S_fix, r_fix, rv_fix, ma5_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                vol_mean.append(info["model"].predict(scaler.transform(X_in))[0]
                                if info["linear"] else info["model"].predict(X_in)[0])
            vol_arr  = np.array(vol_mean)
            vol_lb   = np.maximum(vol_arr + q_low, VOL_FLOOR)
            vol_ub   = vol_arr + q_high
            price_mean = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_arr]).flatten()
            price_lb   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_lb]).flatten()
            price_ub   = np.array([black_scholes_vec(S_fix, STRIKE, T_MATURITY, r_fix, v) for v in vol_ub]).flatten()
            res["dual_sent"][name] = {"vol": vol_arr}
            axes[0,2].plot(joint_shock_label, vol_arr, color=info["color"], linewidth=2.5, label=name)
            axes[0,2].fill_between(joint_shock_label, vol_lb, vol_ub, color=info["color"], alpha=0.15)
            axes[1,2].plot(joint_shock_label, price_mean, color=info["color"], linewidth=2.5)
            axes[1,2].fill_between(joint_shock_label, price_lb, price_ub, color=info["color"], alpha=0.15)
            dvol = np.gradient(vol_arr, joint_shock_label)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(joint_shock_label) > X_THRESHOLD)
                elas = np.where(mask, dvol * (joint_shock_label / vol_arr), np.nan)
            axes[2,2].plot(joint_shock_label, elas, color=info["color"], linewidth=2)

        axes[0,0].set_title("Volatility | Sent_Lag1 (±0.5σ)"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        axes[1,0].set_title("Option Price | Sent_Lag1");        axes[1,0].grid(alpha=0.3)
        axes[2,0].set_title("Elasticity | Sent_Lag1");          axes[2,0].axhline(0,c='k',ls='--'); axes[2,0].grid(alpha=0.3)
        axes[0,1].set_title("Volatility | Sent_Ma5 (±0.5σ)");  axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
        axes[1,1].set_title("Option Price | Sent_Ma5");         axes[1,1].grid(alpha=0.3)
        axes[2,1].set_title("Elasticity | Sent_Ma5");           axes[2,1].axhline(0,c='k',ls='--'); axes[2,1].grid(alpha=0.3)
        axes[0,2].set_title("Volatility | Joint Shock (×σ)");  axes[0,2].legend(); axes[0,2].grid(alpha=0.3)
        axes[1,2].set_title("Option Price | Joint Shock");      axes[1,2].grid(alpha=0.3)
        axes[2,2].set_title("Elasticity | Joint Shock")
        axes[2,2].set_xlabel("Joint Sentiment Shock (×Std)")
        axes[2,2].axhline(0,c='k',ls='--'); axes[2,2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"plots/sensitivity_final_{regime}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("\n" + "="*100)
        print(f"[{regime}] 平均弹性结果（统一阈值：≥0.05σ）")
        print("="*100)
        for name in model_info.keys():
            vol1  = res["sent_lag1"][name]["vol"]
            dvol1 = np.gradient(vol1, sent_lag1_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                m1   = (np.abs(vol1) > 1e-6) & (np.abs(sent_lag1_x-lag1_mu)/lag1_std > X_THRESHOLD)
                elas1 = np.nanmean(np.where(m1, dvol1*(sent_lag1_x/vol1), np.nan))
            vol2  = res["sent_ma5"][name]["vol"]
            dvol2 = np.gradient(vol2, sent_ma5_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                m2   = (np.abs(vol2) > 1e-6) & (np.abs(sent_ma5_x-ma5_mu)/ma5_std > X_THRESHOLD)
                elas2 = np.nanmean(np.where(m2, dvol2*(sent_ma5_x/vol2), np.nan))
            vol3  = res["dual_sent"][name]["vol"]
            dvol3 = np.gradient(vol3, joint_shock_label)
            with np.errstate(divide='ignore', invalid='ignore'):
                m3   = (np.abs(vol3) > 1e-6) & (np.abs(joint_shock_label) > X_THRESHOLD)
                elas3 = np.nanmean(np.where(m3, dvol3*(joint_shock_label/vol3), np.nan))
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
    vega  = np.where(invalid, np.nan, S * norm.pdf(d1) * np.sqrt(T))
    theta_g = np.where(invalid, np.nan,
                       -(S * norm.pdf(d1) * sigma_safe) / (2 * np.sqrt(T))
                       - r * K * np.exp(-r*T) * norm.cdf(d2))
    rho_g = np.where(invalid, np.nan, K * T * np.exp(-r*T) * norm.cdf(d2))
    return delta, gamma, vega, theta_g, rho_g

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

# ===================== 显著性检验 =====================
from scipy.stats import mannwhitneyu
model_config = {"Linear Regression (Ridge)": "lr_pred",
                "Random Forest": "rf_pred",
                "XGBoost": "xgb_pred"}
low_mask  = df_test["regime"] == "Low_Vol"
high_mask = df_test["regime"] == "High_Vol"

print("\n" + "="*70)
print("三个模型 - 高低波动误差差异显著性检验 (Mann-Whitney U)")
print("="*70)
significant_models = []
for model_name, pred_col in model_config.items():
    err_low  = np.abs(y_test[low_mask]  - df_test.loc[low_mask,  pred_col])
    err_high = np.abs(y_test[high_mask] - df_test.loc[high_mask, pred_col])
    stat, p_val = mannwhitneyu(err_low, err_high, alternative="two-sided")
    is_sig = p_val < 0.05
    if is_sig:
        significant_models.append(model_name)
    print(f"【{model_name}】p值 = {p_val:.4f} | 差异：{'统计显著' if is_sig else '不显著'}")

print(f"\n检验总结：{', '.join(significant_models) if significant_models else '无'} 模型呈现显著差异")
print("\n全部任务完成！")