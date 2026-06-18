import pandas as pd
import numpy as np
import os
import random
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.stats import norm
from scipy import stats
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ===================== 全局配置 =====================
# 基础随机种子
SEED = 42

# 交易与窗口参数
TRADING_DAYS = 252
ROLLING_WINDOWS = [5, 10, 20, 60]
GAP_DAYS = 5
TSCV_SPLITS = 3

# 期权定价核心参数
STRIKE = 110
T_MATURITY = 1 / 12
EPS = 1e-8
DIR_THRESHOLD = 1e-4

# 数据集划分参数
TEST_SIZE = 0.5
VAL_SIZE_RATIO = 0.3

# 模型超参数网格
## 线性模型（Ridge）
RIDGE_PARAM_GRID = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
    "ridge__random_state": [SEED]
}

## 随机森林
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 3],
    "random_state": [SEED]
}

## XGBoost（波动率预测）
XGB_PARAM_DIST = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [5, 10],
    "random_state": [SEED]
}
XGB_N_ITER = 10

## XGBoost（E2E定价）
E2E_XGB_PARAM_DIST = XGB_PARAM_DIST.copy()
E2E_XGB_N_ITER = 10

# Heston校准参数网格
HESTON_CALIB_GRID = {
    "kappa": [0.5, 1.0, 2.0, 4.0],
    "theta": [0.02, 0.04, 0.06],
    "xi": [0.2, 0.5, 0.8],
    "rho": [-0.9, -0.7, -0.5, -0.3],
}

# SABR校准参数网格
SABR_CALIB_GRID = {
    "beta": [0.0, 0.5, 1.0],
    "rho": [-0.5, -0.3, 0.0],
    "nu": [0.2, 0.4, 0.6, 0.8],
}

# 希腊字母数值差分步长
DIFF_EPS = 1e-4

# 绘图配置
FIG_SIZE_STD = (12, 6)
FIG_SIZE_LARGE = (18, 6)
FIG_DPI = 300
PALETTE_VIRIDIS = "viridis"
PALETTE_COOLWARM = "coolwarm"

# 输出路径配置
DIRS = ["models", "plots", "metadata", "reports", "cv_results"]
DATA_PATH = "data/processed_data.csv"

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===================== 全局初始化 =====================
# 设置随机种子
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# 自动创建输出目录
for directory in DIRS:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.error(f"创建目录 {directory} 失败: {str(e)}")
        raise

# 加载数据
try:
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"数据加载成功，共 {len(df)} 条记录")
except Exception as e:
    logger.error(f"数据加载失败: {str(e)}")
    raise

# ===================== 核心函数 =====================
def black_scholes_vec(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    invalid = (sigma <= EPS) | (T <= 0) | np.isnan(S) | np.isnan(r) | np.isnan(sigma)
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

def expanding_window_validation(X, y, model, tscv):
    """
    扩张窗口交叉验证，每个fold独立克隆并训练模型，避免数据泄露
    Args:
        X: 特征数据
        y: 目标变量
        model: 模型实例（支持Pipeline）
        tscv: 时间序列交叉验证分割器
    Returns:
        rmse_list: 每个fold的RMSE
        fold_idx: fold序号列表
    """
    rmse_list = []
    fold_idx = []
    try:
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]
            
            # 克隆模型，每个fold独立训练
            model_fold = clone(model)
            model_fold.fit(X_train_fold, y_train_fold)
            pred = model_fold.predict(X_test_fold)
            
            rmse = np.sqrt(mean_squared_error(y_test_fold, pred))
            rmse_list.append(rmse)
            fold_idx.append(i + 1)
        return rmse_list, fold_idx
    except Exception as e:
        logger.error(f"扩张窗口验证失败: {str(e)}")
        raise

# ===================== 多滚动窗口测试 =====================
all_window_results = []
for ROLLING_WINDOW in ROLLING_WINDOWS:
    logger.info(f"测试滚动窗口：{ROLLING_WINDOW} 交易日")

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

    # 初始化时间序列交叉验证
    try:
        tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS, gap=GAP_DAYS)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)

    # 线性模型：Pipeline封装标准化+Ridge，避免交叉验证数据泄露
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    try:
        lr_search = GridSearchCV(
            lr_pipeline,
            RIDGE_PARAM_GRID,
            cv=tscv,
            scoring="neg_root_mean_squared_error"
        )
        lr_search.fit(X_train, y_train)
        lr_model = lr_search.best_estimator_
    except Exception as e:
        logger.error(f"Ridge模型训练失败: {str(e)}")
        raise

    # 随机森林模型
    try:
        rf_search = GridSearchCV(
            RandomForestRegressor(),
            RF_PARAM_GRID,
            cv=tscv,
            scoring="neg_root_mean_squared_error"
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
    except Exception as e:
        logger.error(f"随机森林模型训练失败: {str(e)}")
        raise

    # XGBoost模型
    try:
        xgb_search = RandomizedSearchCV(
            XGBRegressor(),
            XGB_PARAM_DIST,
            n_iter=XGB_N_ITER,
            cv=tscv,
            random_state=SEED
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
    except Exception as e:
        logger.error(f"XGBoost模型训练失败: {str(e)}")
        raise

    # 测试集预测
    lr_pred = lr_model.predict(X_test)
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
logger.info(f"\n多窗口测试结果已保存：reports/rolling_window_comparison.csv")

# 滚动窗口RMSE对比图
try:
    plt.figure(figsize=FIG_SIZE_STD)
    sns.barplot(data=window_df, x="Rolling_Window", y="RMSE", hue="Model", palette=PALETTE_VIRIDIS)
    plt.title("Rolling Window Impact on Volatility Prediction (RMSE)")
    plt.xlabel("Rolling Window (Trading Days)")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/window_rmse_comparison.png", dpi=FIG_DPI)
    logger.info("窗口RMSE对比图已保存")
except Exception as e:
    logger.error(f"保存窗口RMSE对比图失败: {str(e)}")
finally:
    plt.close()

# 滚动窗口R²对比图
try:
    plt.figure(figsize=FIG_SIZE_STD)
    sns.barplot(data=window_df, x="Rolling_Window", y="R2", hue="Model", palette=PALETTE_COOLWARM)
    plt.title("Rolling Window Impact on Volatility Prediction (R2)")
    plt.xlabel("Rolling Window (Trading Days)")
    plt.ylabel("R2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/window_r2_comparison.png", dpi=FIG_DPI)
    logger.info("窗口R²对比图已保存")
except Exception as e:
    logger.error(f"保存窗口R²对比图失败: {str(e)}")
finally:
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

# 用训练集波动率计算regime阈值，避免未来信息泄露
vol_threshold = np.percentile(df_train["rolling_vol"], 50)
df_train["regime"] = np.where(df_train["rolling_vol"] <= vol_threshold, "Low_Vol", "High_Vol")
df_test["regime"] = np.where(df_test["rolling_vol"] <= vol_threshold, "Low_Vol", "High_Vol")

# 模型训练
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
lr_search = GridSearchCV(lr_pipeline, RIDGE_PARAM_GRID, cv=tscv, scoring="neg_root_mean_squared_error")
lr_search.fit(X_train, y_train)
lr_model = lr_search.best_estimator_

rf_search = GridSearchCV(RandomForestRegressor(), RF_PARAM_GRID, cv=tscv, scoring="neg_root_mean_squared_error")
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

xgb_search = RandomizedSearchCV(XGBRegressor(), XGB_PARAM_DIST, n_iter=XGB_N_ITER, cv=tscv, random_state=SEED)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

lr_pred = lr_model.predict(X_test)
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

# E2E模型超参数搜索，与波动率模型调参力度一致
scaler_e2e = StandardScaler()
X_train_e2e_scaled = scaler_e2e.fit_transform(X_train_e2e)
X_test_e2e_scaled = scaler_e2e.transform(X_test_e2e)

try:
    e2e_xgb_search = RandomizedSearchCV(
        XGBRegressor(),
        E2E_XGB_PARAM_DIST,
        n_iter=E2E_XGB_N_ITER,
        cv=tscv,
        random_state=SEED,
        scoring="neg_root_mean_squared_error"
    )
    e2e_xgb_search.fit(X_train_e2e_scaled, y_train_e2e)
    e2e_model = e2e_xgb_search.best_estimator_
    logger.info(f"E2E模型训练完成，最优参数已保存")
except Exception as e:
    logger.error(f"E2E模型训练失败: {str(e)}")
    raise

e2e_pred = e2e_model.predict(X_test_e2e_scaled)
e2e_mse, e2e_mae, e2e_rmse, e2e_r2, e2e_dir = evaluate(y_test_e2e, e2e_pred)

logger.info("\n" + "="*50)
logger.info("E2E期权定价（R²高为正常，因拟合期权价格）")
logger.info(f"E2E RMSE: {e2e_rmse:.4f}")
logger.info(f"E2E R2: {e2e_r2:.4f}")
logger.info("="*50)

# ===================== 基础可视化 =====================
# 特征相关性热力图
try:
    plt.figure(figsize=(10, 8))
    corr = df_final[FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap=PALETTE_COOLWARM, fmt=".2f")
    plt.title("Feature Correlation Heatmap (Volatility Prediction)")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=FIG_DPI)
    logger.info("特征相关性热力图已保存")
except Exception as e:
    logger.error(f"保存特征相关性热力图失败: {str(e)}")
finally:
    plt.close()

# 波动率预测时序图
try:
    plt.figure(figsize=FIG_SIZE_STD)
    plt.plot(df_test['date'], y_test, label='Actual Volatility', color='#2E86AB')
    plt.plot(df_test['date'], y_pred_vol, label='Predicted Volatility', color='#FF6B6B')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Volatility Forecast vs Actual (Volatility Prediction)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/vol_prediction.png", dpi=FIG_DPI)
    logger.info("波动率预测时序图已保存")
except Exception as e:
    logger.error(f"保存波动率预测时序图失败: {str(e)}")
finally:
    plt.close()

# 扩张窗口验证图
expanding_rmse_list, fold_idx = expanding_window_validation(X_train, y_train, best_model, tscv)
try:
    plt.figure(figsize=(10, 6))
    plt.plot(fold_idx, expanding_rmse_list, marker='o', color='#2E86AB')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('Expanding Window Validation RMSE (Volatility Prediction)')
    plt.tight_layout()
    plt.savefig("plots/expanding_window_validation.png", dpi=FIG_DPI)
    logger.info("扩张窗口验证图已保存")
except Exception as e:
    logger.error(f"保存扩张窗口验证图失败: {str(e)}")
finally:
    plt.close()

# ===================== 全模型分Regime性能评估 =====================
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
logger.info("\n全模型分市场状态报告已保存：reports/regime_all_models_results.csv")

logger.info("\n" + "="*60)
logger.info("全模型分市场状态(Regime)性能总览（Volatility Prediction）")
logger.info("="*60)
logger.info(regime_all_df.round(4).to_string(index=False))

# Regime RMSE对比图
try:
    plt.figure(figsize=FIG_SIZE_STD)
    sns.barplot(data=regime_all_df, x="Regime", y="RMSE", hue="Model", palette=PALETTE_VIRIDIS)
    plt.title("All Models Performance Across Volatility Regimes (RMSE)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/regime_all_models_rmse.png", dpi=FIG_DPI)
    logger.info("分Regime RMSE对比图已保存")
except Exception as e:
    logger.error(f"保存分Regime RMSE对比图失败: {str(e)}")
finally:
    plt.close()

# Regime R²对比图
try:
    plt.figure(figsize=FIG_SIZE_STD)
    sns.barplot(data=regime_all_df, x="Regime", y="R2", hue="Model", palette=PALETTE_COOLWARM)
    plt.title("All Models Performance Across Volatility Regimes (R2)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/regime_all_models_r2.png", dpi=FIG_DPI)
    logger.info("分Regime R²对比图已保存")
except Exception as e:
    logger.error(f"保存分Regime R²对比图失败: {str(e)}")
finally:
    plt.close()

# ===================== Parity Plot =====================
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_pred
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = rf_pred
xgb_train_pred = best_xgb.predict(X_train)
xgb_test_pred = xgb_pred

try:
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_LARGE, sharex=True, sharey=True)
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
    plt.savefig("plots/model_performance_parity_all.png", dpi=FIG_DPI, bbox_inches='tight')
    logger.info("Parity Plot已保存")
except Exception as e:
    logger.error(f"保存Parity Plot失败: {str(e)}")
finally:
    plt.close()

# ===================== SHAP分析 =====================
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']
model_list = [lr_model, best_rf, best_xgb]
model_names_shap = ["LR_Ridge", "RandomForest", "XGBoost"]
is_linear_list = [True, False, False]
regimes = ["Low_Vol", "High_Vol"]

def plot_standardized_coefficients(model, features):
    coef = model.named_steps['ridge'].coef_
    coef_df = pd.DataFrame({"Feature": features, "Standardized_Coefficient": coef})
    coef_df["Abs_Coeff"] = coef_df["Standardized_Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coeff", ascending=False).reset_index(drop=True)
    coef_df.to_csv("reports/standardized_coefficients.csv", index=False)
    logger.info("\n" + "="*65)
    logger.info("Linear Regression (Ridge) 标准化回归系数")
    logger.info("="*65)
    logger.info(coef_df[["Feature", "Standardized_Coefficient"]].round(4).to_string(index=False))
    
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Standardized_Coefficient", y="Feature", data=coef_df, palette=PALETTE_COOLWARM)
        plt.title("Ridge Model - Standardized Regression Coefficients", fontsize=14)
        plt.xlabel("Standardized Coefficient (Feature Impact Size)")
        plt.ylabel("Input Feature")
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("plots/standardized_coefficients.png", dpi=FIG_DPI)
        logger.info("标准化系数图已保存")
    except Exception as e:
        logger.error(f"保存标准化系数图失败: {str(e)}")
    finally:
        plt.close()
    return coef_df

standardized_coef_df = plot_standardized_coefficients(lr_model, FEATURES)

def run_regime_shap(model, model_name, is_linear, X_test, df_test, scaler=None):
    for regime in regimes:
        mask = df_test["regime"] == regime
        if mask.sum() < 5:
            continue
        X_reg = X_test[mask].copy()
        if is_linear:
            X_reg_scaled = model.named_steps['scaler'].transform(X_reg)
            explainer = shap.LinearExplainer(model.named_steps['ridge'], X_reg_scaled)
            sv = explainer.shap_values(X_reg_scaled)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_reg)
        
        try:
            plt.figure()
            shap.summary_plot(sv, X_reg, plot_type="bar", show=False)
            plt.title(f"{model_name} | {regime} | Feature Importance (Volatility)")
            plt.tight_layout()
            plt.savefig(f"plots/shap_{model_name}_{regime}_importance.png", bbox_inches='tight', dpi=FIG_DPI)
            logger.info(f"{model_name} {regime} SHAP重要性图已保存")
        except Exception as e:
            logger.error(f"生成{model_name} {regime} SHAP图失败: {str(e)}")
        finally:
            plt.close()

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_regime_shap(model, name, linear, X_test, df_test)

if best_model_name != "Linear Regression":
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
        plt.tight_layout()
        plt.savefig("plots/shap_beeswarm.png", dpi=FIG_DPI, bbox_inches='tight')
        logger.info("最优模型SHAP小提琴图已保存")
    except Exception as e:
        logger.error(f"生成SHAP小提琴图失败: {str(e)}")
    finally:
        plt.close()

def run_full_interaction_shap(model, model_name, is_linear, X_test, scaler=None):
    if is_linear:
        X_scaled = model.named_steps['scaler'].transform(X_test)
        explainer = shap.LinearExplainer(model.named_steps['ridge'], X_scaled)
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
            plt.savefig(f"plots/shap_interact_{model_name}_{feat1}_{feat2}.png", dpi=FIG_DPI, bbox_inches='tight')
        except Exception as e:
            logger.error(f"[{model_name}] 生成 {feat1}-{feat2} 交互图失败: {str(e)}")
        finally:
            plt.close()

for model, name, linear in zip(model_list, model_names_shap, is_linear_list):
    run_full_interaction_shap(model, name, linear, X_test)

# ===================== 双重定价 + 置信区间修正 =====================
df_test['pred_vol'] = y_pred_vol
df_test['two_step_price'] = black_scholes_vec(df_test['S'], STRIKE, T_MATURITY, df_test['r'], df_test['pred_vol'])

# 用验证集残差计算经验分位数置信区间，分Regime
val_size = int(len(df_train) * VAL_SIZE_RATIO)
df_val = df_train.iloc[-val_size:].copy()
X_val = df_val[FEATURES]
y_val = df_val['target_vol_t1']
S_val = df_val['S'].values
r_val = df_val['r'].values

# 验证集预测
if best_model_name == "Linear Regression":
    pred_vol_val = lr_model.predict(X_val)
elif best_model_name == "Random Forest":
    pred_vol_val = best_rf.predict(X_val)
else:
    pred_vol_val = best_xgb.predict(X_val)

true_price_val = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val.values)
two_step_price_val = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, pred_vol_val)
residuals_val = true_price_val - two_step_price_val

# 分Regime计算经验分位数
price_ci = {}
for regime in ["Low_Vol", "High_Vol"]:
    mask = df_val["regime"] == regime
    if mask.sum() < 10:
        logger.warning(f"验证集 {regime} 样本量不足，使用全量残差分位数")
        continue
    res_reg = residuals_val[mask]
    price_ci[regime] = {
        "lower": np.percentile(res_reg, 2.5),
        "upper": np.percentile(res_reg, 97.5)
    }

# 兜底：全量残差分位数
global_lower = np.percentile(residuals_val, 2.5)
global_upper = np.percentile(residuals_val, 97.5)
for regime in ["Low_Vol", "High_Vol"]:
    if regime not in price_ci:
        price_ci[regime] = {"lower": global_lower, "upper": global_upper}

# 测试集分Regime生成置信区间
df_test['two_step_price_lower'] = np.nan
df_test['two_step_price_upper'] = np.nan
for regime in ["Low_Vol", "High_Vol"]:
    mask = df_test["regime"] == regime
    df_test.loc[mask, "two_step_price_lower"] = df_test.loc[mask, "two_step_price"] + price_ci[regime]["lower"]
    df_test.loc[mask, "two_step_price_upper"] = df_test.loc[mask, "two_step_price"] + price_ci[regime]["upper"]

pricing_df = df_test[['date', 'two_step_price', 'two_step_price_lower', 'two_step_price_upper', 'bsm_baseline']].copy()
pricing_df['e2e_price'] = e2e_pred
pricing_df.to_csv("reports/dual_pricing_with_95CI.csv", index=False)
logger.info("双重定价置信区间结果已保存")

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

def heston_price_vec(S_arr, K, T, r_arr, pred_vol_arr, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7):
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

def sabr_price_vec(S_arr, K, T, r_arr, pred_vol_arr, beta=0.5, rho=-0.3, nu=0.4):
    prices = []
    for S, r, alpha in zip(S_arr, r_arr, pred_vol_arr):
        F     = S * np.exp(r * T)
        alpha_safe = max(alpha, 1e-4)
        sigma_sabr = sabr_implied_vol(F, K, T, alpha_safe, beta, rho, nu)
        if np.isnan(sigma_sabr) or sigma_sabr <= 0:
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
# ============================================================
def calibrate_heston(df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate):
    S_val   = df_val['S'].values
    r_val   = df_val['r'].values
    vol_val = df_val['rv_lag1'].values
    y_val   = df_val['target_vol_t1'].values
    true_p  = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val)

    param_grid = HESTON_CALIB_GRID
    best_rmse   = np.inf
    best_params = {'kappa': 2.0, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7}
    records     = []

    total = (len(param_grid['kappa']) * len(param_grid['theta'])
             * len(param_grid['xi'])  * len(param_grid['rho']))
    logger.info(f"\nHeston grid search: {total} 组合...")

    for kappa in param_grid['kappa']:
        for theta in param_grid['theta']:
            for xi in param_grid['xi']:
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

    logger.info(f"Heston 最优参数：{best_params}  |  Val RMSE={best_rmse:.4f}")
    logger.info(f"Feller 条件验证：2κθ - ξ² = {2*best_params['kappa']*best_params['theta'] - best_params['xi']**2:.4f} > 0 ✓")
    return best_params, calib_df

def calibrate_sabr(df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate):
    S_val   = df_val['S'].values
    r_val   = df_val['r'].values
    vol_val = df_val['rv_lag1'].values
    y_val   = df_val['target_vol_t1'].values
    true_p  = black_scholes_vec(S_val, STRIKE, T_MATURITY, r_val, y_val)

    param_grid = SABR_CALIB_GRID
    best_rmse   = np.inf
    best_params = {'beta': 0.5, 'rho': -0.3, 'nu': 0.4}
    records     = []

    total = len(param_grid['beta']) * len(param_grid['rho']) * len(param_grid['nu'])
    logger.info(f"\nSABR grid search: {total} 组合...")

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

    logger.info(f"SABR 最优参数：{best_params}  |  Val RMSE={best_rmse:.4f}")
    return best_params, calib_df

# ============================================================
# 新增模块：三模型定价主流水线
# ============================================================
def run_three_model_pricing(df_test, STRIKE, T_MATURITY, y_pred_vol, y_test,
                            black_scholes_vec, evaluate,
                            heston_params, sabr_params,
                            ensemble_weights_low=(0.5, 0.3, 0.2),
                            ensemble_weights_high=(0.2, 0.3, 0.5)):
    S_arr      = df_test['S'].values
    r_arr      = df_test['r'].values
    regime_arr = df_test['regime'].values

    logger.info("\n" + "="*60)
    logger.info("三模型定价计算中（使用校准参数）...")
    logger.info(f"  Heston: {heston_params}")
    logger.info(f"  SABR:   {sabr_params}")
    logger.info("="*60)

    bsm_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol)
    heston_prices = heston_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **heston_params)
    sabr_prices = sabr_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **sabr_params)
    ensemble_prices = ensemble_price(bsm_prices, heston_prices, sabr_prices, regime_arr,
                                      weights_low=ensemble_weights_low, weights_high=ensemble_weights_high)

    true_bsm_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr, np.asarray(y_test))

    logger.info("\n" + "="*60)
    logger.info("三模型定价性能对比（benchmark = true BSM price）")
    logger.info("="*60)

    pricing_results = []
    for name, pred_p in [("BSM", bsm_prices), ("Heston", heston_prices),
                          ("SABR", sabr_prices), ("Ensemble", ensemble_prices)]:
        valid = np.isfinite(pred_p) & np.isfinite(true_bsm_prices)
        if valid.sum() == 0:
            logger.info(f"{name}: 无有效预测")
            continue
        mse, mae, rmse, r2, _ = evaluate(true_bsm_prices[valid], pred_p[valid])
        pricing_results.append([name, round(rmse,4), round(mae,4), round(r2,4)])
        logger.info(f"{name:10s} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    pd.DataFrame(pricing_results, columns=["Model","RMSE","MAE","R2"]).to_csv(
        "reports/three_model_pricing_comparison.csv", index=False)

    logger.info("\n" + "="*60)
    logger.info("分 Regime 定价性能对比")
    logger.info("="*60)

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
            logger.info(f"{regime:10s} | {name:10s} | N={mask.sum()} | RMSE={rmse:.4f} | R²={r2:.4f}")

    pd.DataFrame(regime_pricing_results, columns=["Regime","Model","N","RMSE","R2"]).to_csv(
        "reports/regime_three_model_pricing.csv", index=False)

    output_df = df_test[['date','regime']].copy()
    output_df['pred_vol']       = y_pred_vol
    output_df['true_bsm_price'] = true_bsm_prices
    output_df['bsm_price']      = bsm_prices
    output_df['heston_price']   = heston_prices
    output_df['sabr_price']     = sabr_prices
    output_df['ensemble_price']  = ensemble_prices
    output_df.to_csv("reports/three_model_pricing_full.csv", index=False)

    # 时序图
    try:
        plt.figure(figsize=(14, 6))
        dates = df_test['date'].values
        plt.plot(dates, true_bsm_prices, label='True BSM Price',  color='black',   linewidth=1.5, alpha=0.8)
        plt.plot(dates, bsm_prices,      label='BSM (pred vol)',  color='#378ADD', linewidth=1.2, linestyle='--')
        plt.plot(dates, heston_prices,   label='Heston',          color='#1D9E75', linewidth=1.2, linestyle='-.')
        plt.plot(dates, sabr_prices,     label='SABR',            color='#BA7517', linewidth=1.2, linestyle=':')
        plt.plot(dates, ensemble_prices, label='Ensemble',        color='#D4537E', linewidth=2.0)
        plt.xlabel('Date')
        plt.ylabel('Option Price')
        plt.title('Three-Model Option Pricing Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/three_model_pricing_timeseries.png", dpi=FIG_DPI)
        logger.info("三模型定价时序图已保存")
    except Exception as e:
        logger.error(f"保存三模型定价时序图失败: {str(e)}")
    finally:
        plt.close()

    # Regime RMSE条形图
    try:
        if regime_pricing_results:
            rpr_df = pd.DataFrame(regime_pricing_results, columns=["Regime","Model","N","RMSE","R2"])
            plt.figure(figsize=(10, 5))
            sns.barplot(data=rpr_df, x="Regime", y="RMSE", hue="Model",
                        palette=["#378ADD","#1D9E75","#BA7517","#D4537E"])
            plt.title("Three-Model Pricing RMSE by Volatility Regime")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/three_model_regime_rmse.png", dpi=FIG_DPI)
            logger.info("分Regime三模型RMSE对比图已保存")
    except Exception as e:
        logger.error(f"保存分Regime三模型RMSE图失败: {str(e)}")
    finally:
        plt.close()

    return output_df

# ============================================================
# 新增模块：希腊字母计算（BSM / Heston / SABR）
# ============================================================
def calculate_greeks(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    invalid = (sigma <= 1e-6) | (T <= 0) | np.isnan(S) | np.isnan(r) | np.isnan(sigma)
    sigma_safe = np.where(invalid, 1.0, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * np.sqrt(T))
    d2 = d1 - sigma_safe * np.sqrt(T)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma_safe * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma_safe) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
    rho   = K * T * np.exp(-r*T) * norm.cdf(d2)
    
    return np.where(invalid, np.nan, delta), np.where(invalid, np.nan, gamma), \
           np.where(invalid, np.nan, vega), np.where(invalid, np.nan, theta), \
           np.where(invalid, np.nan, rho)

def calculate_heston_greeks_vec(S_arr, K, T, r_arr, sigma_arr, kappa, theta, xi, rho, eps=1e-4):
    delta, gamma, vega, theta_greek, rho_greek = [], [], [], [], []
    for S, r, sigma in zip(S_arr, r_arr, sigma_arr):
        p0 = heston_price_single(S, K, T, r, sigma**2, kappa, theta, xi, rho)
        p_up_s = heston_price_single(S*(1+eps), K, T, r, sigma**2, kappa, theta, xi, rho)
        p_down_s = heston_price_single(S*(1-eps), K, T, r, sigma**2, kappa, theta, xi, rho)
        p_up_v = heston_price_single(S, K, T, r, (sigma*(1+eps))**2, kappa, theta, xi, rho)
        p_down_v = heston_price_single(S, K, T, r, (sigma*(1-eps))**2, kappa, theta, xi, rho)
        p_up_t = heston_price_single(S, K, T*(1+eps), r, sigma**2, kappa, theta, xi, rho)
        p_down_t = heston_price_single(S, K, T*(1-eps), r, sigma**2, kappa, theta, xi, rho)
        p_up_r = heston_price_single(S, K, T, r*(1+eps), sigma**2, kappa, theta, xi, rho)
        p_down_r = heston_price_single(S, K, T, r*(1-eps), sigma**2, kappa, theta, xi, rho)
        
        d = (p_up_s - p_down_s) / (2 * S * eps)
        g = (p_up_s - 2*p0 + p_down_s) / ((S * eps)**2)
        v = (p_up_v - p_down_v) / (2 * sigma * eps)
        th = -(p_up_t - p_down_t) / (2 * T * eps)
        rh = (p_up_r - p_down_r) / (2 * r * eps)
        
        delta.append(d)
        gamma.append(g)
        vega.append(v)
        theta_greek.append(th)
        rho_greek.append(rh)
    return np.array(delta), np.array(gamma), np.array(vega), np.array(theta_greek), np.array(rho_greek)

def calculate_sabr_greeks_vec(S_arr, K, T, r_arr, alpha_arr, beta, rho, nu, eps=1e-4):
    delta, gamma, vega, theta_greek, rho_greek = [], [], [], [], []
    for S, r, alpha in zip(S_arr, r_arr, alpha_arr):
        p0 = sabr_price_single_wrapper(S, K, T, r, alpha, beta, rho, nu)
        p_up_s = sabr_price_single_wrapper(S*(1+eps), K, T, r, alpha, beta, rho, nu)
        p_down_s = sabr_price_single_wrapper(S*(1-eps), K, T, r, alpha, beta, rho, nu)
        p_up_v = sabr_price_single_wrapper(S, K, T, r, alpha*(1+eps), beta, rho, nu)
        p_down_v = sabr_price_single_wrapper(S, K, T, r, alpha*(1-eps), beta, rho, nu)
        p_up_t = sabr_price_single_wrapper(S, K, T*(1+eps), r, alpha, beta, rho, nu)
        p_down_t = sabr_price_single_wrapper(S, K, T*(1-eps), r, alpha, beta, rho, nu)
        p_up_r = sabr_price_single_wrapper(S, K, T, r*(1+eps), alpha, beta, rho, nu)
        p_down_r = sabr_price_single_wrapper(S, K, T, r*(1-eps), alpha, beta, rho, nu)
        
        d = (p_up_s - p_down_s) / (2 * S * eps)
        g = (p_up_s - 2*p0 + p_down_s) / ((S * eps)**2)
        v = (p_up_v - p_down_v) / (2 * alpha * eps)
        th = -(p_up_t - p_down_t) / (2 * T * eps)
        rh = (p_up_r - p_down_r) / (2 * r * eps)
        
        delta.append(d)
        gamma.append(g)
        vega.append(v)
        theta_greek.append(th)
        rho_greek.append(rh)
    return np.array(delta), np.array(gamma), np.array(vega), np.array(theta_greek), np.array(rho_greek)

# ============================================================
# 执行校准 + 三模型定价 + 希腊字母 + 敏感性分析
# ============================================================
best_heston_params, heston_calib_df = calibrate_heston(
    df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate)
best_sabr_params, sabr_calib_df = calibrate_sabr(
    df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate)

# OOS权重推导
def derive_oos_weights(df_val, STRIKE, T_MATURITY, black_scholes_vec, evaluate, heston_params, sabr_params):
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
        _, _, rmse, _, _ = evaluate(y_true=true_p[valid], y_pred=preds[valid])
        rmse_results[name] = rmse

    logger.info("\n" + "="*50)
    logger.info("Validation Set RMSE（用于推导 OOS 权重）")
    logger.info("="*50)
    for name, rmse in rmse_results.items():
        logger.info(f"  {name:8s}: RMSE = {rmse:.4f}")

    inv_rmse   = {k: 1.0/v if v > 0 else 0 for k, v in rmse_results.items()}
    total      = sum(inv_rmse.values())
    oos_w      = {k: v/total for k, v in inv_rmse.items()}

    logger.info(f"\nOOS 权重：BSM={oos_w['BSM']:.3f} | Heston={oos_w['Heston']:.3f} | SABR={oos_w['SABR']:.3f}")
    w_tuple = (oos_w['BSM'], oos_w['Heston'], oos_w['SABR'])
    return w_tuple, rmse_results


# ------------------------------
# SABR 单条定价辅助函数（用于希腊字母数值差分计算）
# ------------------------------
def sabr_price_single_wrapper(S, K, T, r, alpha, beta, rho, nu):
    if T <= 0 or S <= 0 or K <= 0 or alpha <= 0:
        return np.nan
    F = S * np.exp(r * T)
    sigma_sabr = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    if np.isnan(sigma_sabr) or sigma_sabr <= 0:
        return np.nan
    return black_scholes_call(S, K, T, r, sigma_sabr)


# ------------------------------
# Heston 参数敏感性分析
# ------------------------------
def heston_sensitivity(df_test, STRIKE, T_MATURITY, y_pred_vol,
                       black_scholes_vec, evaluate, best_heston_params):
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
    base = best_heston_params.copy()
    records = []

    for param_name, values in param_grid.items():
        for val in values:
            params = base.copy()
            params[param_name] = val
            # Feller 条件硬约束
            if 2 * params['kappa'] * params['theta'] <= params['xi'] ** 2:
                continue
            preds = heston_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **params)
            valid = np.isfinite(preds) & np.isfinite(y_true_prices)
            if valid.sum() == 0:
                continue
            _, _, rmse, r2, _ = evaluate(y_true= y_true_prices[valid], y_pred= preds[valid])
            records.append({
                'param': param_name, 
                'value': val,
                'RMSE': round(rmse,4), 
                'R2': round(r2,4),
                'is_best': (val == base[param_name])
            })

    sens_df = pd.DataFrame(records)
    sens_df.to_csv("reports/heston_sensitivity.csv", index=False)

    try:
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
            ax.set_xlabel(pname)
            ax.set_ylabel('RMSE')
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/heston_sensitivity.png", dpi=FIG_DPI)
        plt.close()
        logger.info("Heston 敏感性分析完成 → plots/heston_sensitivity.png")
    except Exception as e:
        logger.error(f"生成 Heston 敏感性分析图失败: {str(e)}")
    
    return sens_df


# ------------------------------
# SABR 参数敏感性分析
# ------------------------------
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
    base = best_sabr_params.copy()
    records = []

    for param_name, values in param_grid.items():
        for val in values:
            params = base.copy()
            params[param_name] = val
            preds = sabr_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **params)
            valid = np.isfinite(preds) & np.isfinite(y_true_prices)
            if valid.sum() == 0:
                continue
            _, _, rmse, r2, _ = evaluate(y_true= y_true_prices[valid], y_pred= preds[valid])
            records.append({
                'param': param_name, 
                'value': val,
                'RMSE': round(rmse,4), 
                'R2': round(r2,4),
                'is_best': (val == base[param_name])
            })

    sens_df = pd.DataFrame(records)
    sens_df.to_csv("reports/sabr_sensitivity.csv", index=False)

    try:
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
            ax.set_xlabel(pname)
            ax.set_ylabel('RMSE')
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/sabr_sensitivity.png", dpi=FIG_DPI)
        plt.close()
        logger.info("SABR 敏感性分析完成 → plots/sabr_sensitivity.png")
    except Exception as e:
        logger.error(f"生成 SABR 敏感性分析图失败: {str(e)}")
    
    return sens_df


# ------------------------------
# 集成权重敏感性分析
# ------------------------------
def ensemble_weight_sensitivity(df_test, STRIKE, T_MATURITY, y_pred_vol,
                                 black_scholes_vec, evaluate,
                                 heston_params, sabr_params,
                                 oos_weights=None):
    S_arr = df_test['S'].values
    r_arr = df_test['r'].values
    regime_arr = df_test['regime'].values
    y_true_prices = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr,
                                       np.asarray(df_test['target_vol_t1']))

    bsm_p = black_scholes_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol)
    heston_p = heston_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **heston_params)
    sabr_p = sabr_price_vec(S_arr, STRIKE, T_MATURITY, r_arr, y_pred_vol, **sabr_params)

    weight_schemes = {
        "Equal (1/3 each)":      {"low": (1/3,1/3,1/3),   "high": (1/3,1/3,1/3)},
        "Heuristic (BSM-heavy)": {"low": (0.5,0.3,0.2),   "high": (0.2,0.3,0.5)},
        "Pure BSM":              {"low": (1.0,0.0,0.0),   "high": (1.0,0.0,0.0)},
        "Pure Heston":           {"low": (0.0,1.0,0.0),   "high": (0.0,1.0,0.0)},
        "Pure SABR":             {"low": (0.0,0.0,1.0),   "high": (0.0,0.0,1.0)},
    }
    if oos_weights is not None:
        weight_schemes["OOS-derived"] = {"low": oos_weights, "high": oos_weights}

    records = []
    for scheme_name, w in weight_schemes.items():
        ens = ensemble_price(bsm_p, heston_p, sabr_p, regime_arr,
                             weights_low=w["low"], weights_high=w["high"])
        valid = np.isfinite(ens) & np.isfinite(y_true_prices)
        if valid.sum() == 0:
            continue
        _, _, rmse, r2, _ = evaluate(y_true= y_true_prices[valid], y_pred= ens[valid])
        records.append({
            "Weight Scheme": scheme_name,
            "RMSE": round(rmse,4), 
            "R2": round(r2,4)
        })

    weight_df = pd.DataFrame(records).sort_values("RMSE")
    weight_df.to_csv("reports/ensemble_weight_sensitivity.csv", index=False)

    logger.info("\n" + "="*60)
    logger.info("Ensemble 权重敏感性分析")
    logger.info("="*60)
    logger.info(weight_df.to_string(index=False))

    try:
        plt.figure(figsize=(10, 5))
        colors = ['#E24B4A' if i == 0 else '#B5D4F4' for i in range(len(weight_df))]
        plt.barh(weight_df["Weight Scheme"], weight_df["RMSE"], color=colors)
        plt.xlabel('RMSE (lower is better)')
        plt.title('Ensemble Weight Sensitivity\n(red = best scheme)')
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("plots/ensemble_weight_sensitivity.png", dpi=FIG_DPI)
        plt.close()
    except Exception as e:
        logger.error(f"生成集成权重敏感性图失败: {str(e)}")
    
    return weight_df


# ============================================================
# 主执行流程：校准 → 权重推导 → 三模型定价 → 敏感性分析
# ============================================================
# 1. OOS 权重推导
oos_w, _ = derive_oos_weights(
    df_val, STRIKE, T_MATURITY,
    black_scholes_vec, evaluate,
    best_heston_params, best_sabr_params
)

# 2. 三模型定价主流程
pricing_output = run_three_model_pricing(
    df_test, STRIKE, T_MATURITY, y_pred_vol, y_test,
    black_scholes_vec, evaluate,
    heston_params=best_heston_params,
    sabr_params=best_sabr_params,
    ensemble_weights_low=oos_w,
    ensemble_weights_high=oos_w
)

# 3. 参数敏感性分析
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

# 4. 保存校准参数汇总
calib_summary = pd.DataFrame(
    [{"Model": "Heston", "Parameter": k, "Calibrated_Value": v,
       "Constraint": "Feller: 2κθ>ξ²" if k == "xi" else ""}
     for k, v in best_heston_params.items()] +
    [{"Model": "SABR", "Parameter": k, "Calibrated_Value": v, "Constraint": ""}
     for k, v in best_sabr_params.items()]
)
calib_summary.to_csv("reports/calibrated_parameters.csv", index=False)
logger.info("校准参数汇总已保存 → reports/calibrated_parameters.csv")


# ============================================================
# 希腊字母计算与保存（三模型对比）
# ============================================================
logger.info("正在计算三模型希腊字母...")
S_test = df_test['S'].values
r_test = df_test['r'].values
vol_test = y_pred_vol

# BSM 解析解希腊字母
bsm_delta, bsm_gamma, bsm_vega, bsm_theta, bsm_rho = calculate_greeks(
    S_test, STRIKE, T_MATURITY, r_test, vol_test
)

# Heston 数值差分希腊字母
heston_delta, heston_gamma, heston_vega, heston_theta, heston_rho = calculate_heston_greeks_vec(
    S_test, STRIKE, T_MATURITY, r_test, vol_test, **best_heston_params
)

# SABR 数值差分希腊字母
sabr_delta, sabr_gamma, sabr_vega, sabr_theta, sabr_rho = calculate_sabr_greeks_vec(
    S_test, STRIKE, T_MATURITY, r_test, vol_test, **best_sabr_params
)

# 汇总保存
greeks_full = pd.DataFrame({
    "date": df_test['date'].values,
    "regime": df_test['regime'].values,
    "bsm_delta": bsm_delta, "bsm_gamma": bsm_gamma, "bsm_vega": bsm_vega,
    "bsm_theta": bsm_theta, "bsm_rho": bsm_rho,
    "heston_delta": heston_delta, "heston_gamma": heston_gamma, "heston_vega": heston_vega,
    "heston_theta": heston_theta, "heston_rho": heston_rho,
    "sabr_delta": sabr_delta, "sabr_gamma": sabr_gamma, "sabr_vega": sabr_vega,
    "sabr_theta": sabr_theta, "sabr_rho": sabr_rho,
})
greeks_full.to_csv("reports/full_greeks_comparison.csv", index=False)
logger.info("三模型希腊字母对比结果已保存 → reports/full_greeks_comparison.csv")


# ============================================================
# 残差分析（全维度）
# ============================================================
logger.info("正在生成全维度残差分析...")
residual_df = df_test[['date', 'rolling_vol', 'regime']].copy()
residual_df['actual_vol'] = y_test
residual_df['lr_pred_vol'] = lr_pred
residual_df['rf_pred_vol'] = rf_pred
residual_df['xgb_pred_vol'] = xgb_pred
residual_df['best_vol_pred'] = y_pred_vol
residual_df['two_step_price'] = df_test['two_step_price']
residual_df['e2e_price'] = e2e_pred
residual_df['true_bsm_price'] = black_scholes_vec(
    df_test['S'], STRIKE, T_MATURITY, df_test['r'], y_test
)
residual_df['bsm_baseline_price'] = df_test['bsm_baseline']

residual_df['vol_residual'] = residual_df['actual_vol'] - residual_df['best_vol_pred']
residual_df['two_step_price_residual'] = residual_df['true_bsm_price'] - residual_df['two_step_price']
residual_df['e2e_price_residual'] = residual_df['true_bsm_price'] - residual_df['e2e_price']
residual_df.to_csv("reports/full_residual_analysis.csv", index=False)

try:
    plt.figure(figsize=(10, 6))
    plt.hist(residual_df['vol_residual'], bins=30, alpha=0.7, color='#2E86AB')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Volatility Forecast Residual Distribution')
    plt.tight_layout()
    plt.savefig("plots/vol_residual_hist.png", dpi=FIG_DPI)
    plt.close()
    logger.info("波动率残差分布图已保存 → plots/vol_residual_hist.png")
except Exception as e:
    logger.error(f"生成波动率残差图失败: {str(e)}")


# ============================================================
# 情绪敏感性分析
# ============================================================
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
                logger.warning(f"{regime} 区间 {name} 模型残差样本量不足")
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

            # Sent_Lag1 敏感性
            vol_mean = []
            for x in sent_lag1_x:
                x_clip = np.clip(x, sent_lag1_lower, sent_lag1_upper)
                X_in = pd.DataFrame([[x_clip, S_fix, r_fix, rv_fix, ma5_mu]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                pred = info["model"].predict(X_in)[0]
                vol_mean.append(pred)
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
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_lag1_x - lag1_mu)/lag1_std > X_THRESHOLD)
                elas1 = np.where(mask, dvol * (sent_lag1_x / vol_arr), np.nan)
            axes[2,0].plot(sent_lag1_x, elas1, color=info["color"], linewidth=2)

            # Sent_Ma5 敏感性
            vol_mean = []
            for x in sent_ma5_x:
                x_clip = np.clip(x, sent_ma5_lower, sent_ma5_upper)
                X_in = pd.DataFrame([[lag1_mu, S_fix, r_fix, rv_fix, x_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                pred = info["model"].predict(X_in)[0]
                vol_mean.append(pred)
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
                mask = (np.abs(vol_arr) > 1e-6) & (np.abs(sent_ma5_x - ma5_mu)/ma5_std > X_THRESHOLD)
                elas2 = np.where(mask, dvol * (sent_ma5_x / vol_arr), np.nan)
            axes[2,1].plot(sent_ma5_x, elas2, color=info["color"], linewidth=2)

            # 联合冲击敏感性
            vol_mean = []
            for offset in OFFSET_RANGE:
                lag_clip = np.clip(lag1_mu + offset*lag1_std, sent_lag1_lower, sent_lag1_upper)
                ma5_clip = np.clip(ma5_mu  + offset*ma5_std,  sent_ma5_lower,  sent_ma5_upper)
                X_in = pd.DataFrame([[lag_clip, S_fix, r_fix, rv_fix, ma5_clip]],
                                    columns=["sent_lag1","S_lag1","r_lag1","rv_lag1","sent_ma5"])
                pred = info["model"].predict(X_in)[0]
                vol_mean.append(pred)
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
                elas3 = np.where(mask, dvol * (joint_shock_label / vol_arr), np.nan)
            axes[2,2].plot(joint_shock_label, elas3, color=info["color"], linewidth=2)

        axes[0,0].set_title("Volatility | Sent_Lag1 (±0.5σ)")
        axes[0,0].legend()
        axes[0,0].grid(alpha=0.3)
        axes[1,0].set_title("Option Price | Sent_Lag1")
        axes[1,0].grid(alpha=0.3)
        axes[2,0].set_title("Elasticity | Sent_Lag1")
        axes[2,0].axhline(0,c='k',ls='--')
        axes[2,0].grid(alpha=0.3)

        axes[0,1].set_title("Volatility | Sent_Ma5 (±0.5σ)")
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)
        axes[1,1].set_title("Option Price | Sent_Ma5")
        axes[1,1].grid(alpha=0.3)
        axes[2,1].set_title("Elasticity | Sent_Ma5")
        axes[2,1].axhline(0,c='k',ls='--')
        axes[2,1].grid(alpha=0.3)

        axes[0,2].set_title("Volatility | Joint Shock (×σ)")
        axes[0,2].legend()
        axes[0,2].grid(alpha=0.3)
        axes[1,2].set_title("Option Price | Joint Shock")
        axes[1,2].grid(alpha=0.3)
        axes[2,2].set_title("Elasticity | Joint Shock")
        axes[2,2].set_xlabel("Joint Sentiment Shock (×Std)")
        axes[2,2].axhline(0,c='k',ls='--')
        axes[2,2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"plots/sensitivity_final_{regime}.png", dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"[{regime}] 情绪敏感性分析图已保存 → plots/sensitivity_final_{regime}.png")
        logger.info(f"[{regime}] 平均弹性结果（统一阈值：≥0.05σ）")
        for name in model_info.keys():
            vol1  = res["sent_lag1"][name]["vol"]
            dvol1 = np.gradient(vol1, sent_lag1_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                m1   = (np.abs(vol1) > 1e-6) & (np.abs(sent_lag1_x - lag1_mu)/lag1_std > X_THRESHOLD)
                elas1 = np.nanmean(np.where(m1, dvol1*(sent_lag1_x/vol1), np.nan))
            vol2  = res["sent_ma5"][name]["vol"]
            dvol2 = np.gradient(vol2, sent_ma5_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                m2   = (np.abs(vol2) > 1e-6) & (np.abs(sent_ma5_x - ma5_mu)/ma5_std > X_THRESHOLD)
                elas2 = np.nanmean(np.where(m2, dvol2*(sent_ma5_x/vol2), np.nan))
            vol3  = res["dual_sent"][name]["vol"]
            dvol3 = np.gradient(vol3, joint_shock_label)
            with np.errstate(divide='ignore', invalid='ignore'):
                m3   = (np.abs(vol3) > 1e-6) & (np.abs(joint_shock_label) > X_THRESHOLD)
                elas3 = np.nanmean(np.where(m3, dvol3*(joint_shock_label/vol3), np.nan))
            logger.info(f"  {name:6s} | Lag1={elas1:.4f} | Ma5={elas2:.4f} | Joint={elas3:.4f}")

# 执行情绪敏感性分析
plot_final_uncertainty_sensitivity()


# ============================================================
# 波动率预测模型总表
# ============================================================
vol_summary = []
model_vol_map = [
    ["Linear Regression (Ridge L2)", lr_pred],
    ["Random Forest", rf_pred],
    ["XGBoost", xgb_pred]
]
for name, pred in model_vol_map:
    _, _, rmse, r2, dir_acc = evaluate(y_true= y_test, y_pred= pred)
    cv_rmse_list, _ = expanding_window_validation(
        X_train, y_train,
        lr_model if name=="Linear Regression (Ridge L2)" else best_rf if name=="Random Forest" else best_xgb,
        tscv
    )
    cv_rmse = np.mean(cv_rmse_list)
    vol_summary.append([name, round(rmse,4), round(r2,4), round(dir_acc,4), round(cv_rmse,4)])

_, _, bsm_rmse, bsm_r2, bsm_dir = evaluate(y_true= y_test, y_pred= df_test["rv_lag1"])
vol_summary.append(["BSM Baseline (Lagged Vol)", round(bsm_rmse,4), round(bsm_r2,4), round(bsm_dir,4), "N/A"])

vol_summary_df = pd.DataFrame(vol_summary, columns=[
    "Model", "Volatility_RMSE", "Volatility_R2", "Directional_Acc", "Expanding_Window_CV_RMSE"
])
vol_summary_df.to_csv("reports/volatility_prediction_summary.csv", index=False)

print("\n" + "="*70)
print("波动率预测模型性能对比")
print("="*70)
print(vol_summary_df.to_string(index=False))
print("\n结果已保存 → reports/volatility_prediction_summary.csv")


# ============================================================
# 显著性检验（Mann-Whitney U）
# ============================================================
from scipy.stats import mannwhitneyu

model_config = {
    "Linear Regression (Ridge)": "lr_pred",
    "Random Forest": "rf_pred",
    "XGBoost": "xgb_pred"
}
low_mask  = df_test["regime"] == "Low_Vol"
high_mask = df_test["regime"] == "High_Vol"

print("\n" + "="*70)
print("三模型 - 高低波动区间误差差异显著性检验 (Mann-Whitney U)")
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

if significant_models:
    print(f"\n检验结论：{', '.join(significant_models)} 模型在高低波动区间的预测误差存在显著差异")
else:
    print("\n检验结论：所有模型在高低波动区间的预测误差均无显著差异")

print("\n✅ 全部任务执行完成！所有结果已保存至对应目录。")