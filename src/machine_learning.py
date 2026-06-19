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
import tqdm
from scipy.integrate import simpson
def heston_char_func(u, S, K, T, r, v0, kappa, theta, xi, rho, j):
    """Heston特征函数（核心公式完全保留）"""
    if j == 1:
        u_j = u - 1j
        b_j = kappa - rho * xi
    else:
        u_j = u
        b_j = kappa
    
    a = kappa * theta
    sigma_sq = xi ** 2
    d = np.sqrt((rho * xi * u_j * 1j - b_j) ** 2 - sigma_sq * (2 * b_j * u_j * 1j - u_j ** 2))
    g = (b_j - rho * xi * u_j * 1j + d) / (b_j - rho * xi * u_j * 1j - d)
    
    C = r * u_j * T * 1j + (a / sigma_sq) * (
        (b_j - rho * xi * u_j * 1j + d) * T 
        - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
    )
    D = (b_j - rho * xi * u_j * 1j + d) / sigma_sq * (
        (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
    )
    
    return np.exp(C + D * v0 + 1j * u * np.log(S))

def heston_price_single(S, K, T, r, v0, kappa, theta, xi, rho, n_points=50):
    """
    Heston期权定价（固定网格辛普森积分，速度快、不卡死）
    n_points: 积分网格点数，默认100点，精度<0.01%；需更高精度可调到200
    """
    if T <= 0 or S <= 0 or K <= 0 or v0 <= 0:
        return np.nan
    
    # 积分区间与原quad一致：[1e-8, 20]，覆盖被积函数全部有效区域
    u = np.linspace(1e-8, 20, n_points)
    
    # 计算两个概率项的被积函数
    integrand1 = np.real(heston_char_func(u, S, K, T, r, v0, kappa, theta, xi, rho, 1) / (1j * u))
    integrand2 = np.real(heston_char_func(u, S, K, T, r, v0, kappa, theta, xi, rho, 2) / (1j * u))
    
    # 辛普森数值积分
    P1 = 0.5 + simpson(integrand1, u) / np.pi
    P2 = 0.5 + simpson(integrand2, u) / np.pi
    
    price = S * P1 - K * np.exp(-r * T) * P2
    # 内在价值下界保护（无套利约束）
    return max(price, max(S - K * np.exp(-r * T), 0))


def heston_price_vec(S_arr, K, T, r_arr, pred_vol_arr, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, show_progress=False):
    """向量化Heston定价，可选进度条"""
    try:
        from tqdm import tqdm
        iterator = tqdm(zip(S_arr, r_arr, pred_vol_arr), total=len(S_arr), desc="Heston定价计算中", disable=not show_progress)
    except ImportError:
        iterator = zip(S_arr, r_arr, pred_vol_arr)
    
    prices = []
    for S, r, sigma in iterator:
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
    best_rmse = np.inf
    # 兜底默认参数（保证一定满足Feller条件）
    best_params = {'kappa': 4.0, 'theta': 0.06, 'xi': 0.5, 'rho': -0.7}
    records = []

    total = (len(param_grid['kappa']) * len(param_grid['theta'])
             * len(param_grid['xi']) * len(param_grid['rho']))
    logger.info(f"\nHeston grid search: {total} 组合...")

    for kappa in param_grid['kappa']:
        for theta in param_grid['theta']:
            for xi in param_grid['xi']:
                # Feller条件硬约束
                if 2 * kappa * theta <= xi ** 2:
                    continue
                for rho in param_grid['rho']:
                    preds = heston_price_vec(
                        S_val, STRIKE, T_MATURITY, r_val, vol_val,
                        kappa=kappa, theta=theta, xi=xi, rho=rho
                    )
                    valid = np.isfinite(preds) & np.isfinite(true_p)
                    # 降低有效样本阈值，适配小样本验证集
                    if valid.sum() < 3:
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
                        best_rmse = rmse
                        best_params = {'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho}

    # 空结果保护：如果没有有效组合，使用兜底默认参数
    if not records:
        logger.warning("Heston校准无有效参数组合，使用默认兜底参数")
        calib_df = pd.DataFrame([{**best_params, 'feller': round(2*best_params['kappa']*best_params['theta'] - best_params['xi']**2, 4), 'val_RMSE': np.nan, 'val_R2': np.nan}])
    else:
        calib_df = pd.DataFrame(records).sort_values('val_RMSE')
    
    calib_df.to_csv("reports/heston_calibration_grid.csv", index=False)

    logger.info(f"Heston 最优参数：{best_params}  |  Val RMSE: {best_rmse:.4f}")
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

print("\n全部任务执行完成！所有结果已保存至对应目录。")

# ============================================================
# 新增模块：LSSM波动率预测 + PINN定价引擎 + 分桶尾部风险评估
# ============================================================
# ============================================================
# 新增模块：LSSM波动率预测 + PINN定价引擎 + 分桶尾部风险评估
# 插入位置：现有代码末尾（显著性检验之后）
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pykalman import KalmanFilter

# ===================== 全局参数配置 =====================
# LSSM参数
LSSM_N_COMPONENTS = 4
LSSM_L2_TRANSITION = 0.1

# PINN定价引擎参数
PINN_HIDDEN_DIMS = [64, 32]
PINN_EPOCHS = 200
PINN_LR = 1e-3
PINN_BATCH_SIZE = 32
PINN_LAMBDA_GRID = [0.0, 0.01, 0.1, 1.0]

# 分桶评估参数（低估幅度分桶）
UNDERPRICE_BUCKETS = [0, 0.02, 0.05, 0.10, 1.0]
BUCKET_LABELS = ["0-2%", "2-5%", "5-10%", ">10%"]

# ===================== 模块1：LSSM波动率预测器 =====================
def train_lssm_vol_predictor(X_train, y_train, X_val, y_val, X_test, features,
                               n_components=4, l2_transition=0.1, seed=42):
    """
    线性状态空间模型（Kalman滤波）波动率预测
    架构：特征标准化 → Kalman滤波提取隐状态 → 拼接原始特征 → Ridge读出层
    严格时序：仅用历史信息，无未来泄露
    """
    np.random.seed(seed)
    logger.info(f"训练LSSM波动率预测器 | 隐状态维度={n_components} | L2强度={l2_transition}")

    # Step1：特征标准化
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[features])
    X_val_scaled = scaler.transform(X_val[features])
    X_te_scaled = scaler.transform(X_test[features])

    # Step2：Kalman滤波模型拟合（EM算法估计参数）
    kf = KalmanFilter(
        n_dim_obs=X_tr_scaled.shape[1],
        n_dim_state=n_components,
        transition_matrices=np.eye(n_components) * 0.9,
        observation_matrices=np.random.randn(X_tr_scaled.shape[1], n_components) * 0.1,
        transition_covariance=np.eye(n_components) * l2_transition,
        observation_covariance=np.eye(X_tr_scaled.shape[1]) * 0.1,
        initial_state_mean=np.zeros(n_components),
        initial_state_covariance=np.eye(n_components),
        em_vars=['transition_matrices', 'observation_matrices',
                 'transition_covariance', 'observation_covariance']
    )

    try:
        kf.em(X_tr_scaled, n_iter=20)
    except Exception as e:
        logger.error(f"Kalman滤波EM拟合失败: {str(e)}")
        raise

    # Step3：提取各数据集隐状态（严格滤波，不用平滑）
    states_train, _ = kf.filter(X_tr_scaled)
    states_val, _ = kf.filter(X_val_scaled)
    states_test, _ = kf.filter(X_te_scaled)

    # Step4：拼接隐状态+原始特征，训练Ridge读出层
    X_readout_train = np.hstack([states_train, X_tr_scaled])
    X_readout_val = np.hstack([states_val, X_val_scaled])
    X_readout_test = np.hstack([states_test, X_te_scaled])

    # 验证集选最优alpha
    best_alpha, best_val_rmse = 1.0, np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_readout_train, y_train)
        val_pred = ridge.predict(X_readout_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = alpha

    logger.info(f"LSSM读出层最优alpha={best_alpha} | 验证集RMSE={best_val_rmse:.4f}")

    # 全训练集重训练
    X_readout_full = np.vstack([X_readout_train, X_readout_val])
    y_full = np.concatenate([y_train, y_val])
    final_ridge = Ridge(alpha=best_alpha)
    final_ridge.fit(X_readout_full, y_full)

    # 测试集预测与下界保护
    y_pred_test = final_ridge.predict(X_readout_test)
    y_pred_test = np.maximum(y_pred_test, 1e-4)

    meta = {
        'kf': kf, 'scaler': scaler, 'ridge': final_ridge,
        'best_alpha': best_alpha, 'val_rmse': best_val_rmse
    }
    logger.info(f"LSSM测试集预测完成 | 波动率范围[{y_pred_test.min():.4f}, {y_pred_test.max():.4f}]")
    return y_pred_test, meta

# ===================== 模块2：PINN物理信息定价引擎 =====================
class PricingMLP(nn.Module):
    """期权定价MLP网络：输入定价五因子[S, K, T, r, σ]，输出期权价格"""
    def __init__(self, input_dim=5, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.Tanh()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def bsm_pde_residual(net, S, K, T, r, sigma):
    """计算BSM偏微分方程残差：∂V/∂t + 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0"""
    S = S.requires_grad_(True)
    sigma = sigma.requires_grad_(True)
    T_vec = torch.full_like(S, T).requires_grad_(True)
    K_vec = torch.full_like(S, K)
    r_vec = torch.full_like(S, r)

    x_in = torch.stack([S, K_vec, T_vec, r_vec, sigma], dim=1)
    V = net(x_in)

    # 一阶偏导
    grad_outputs = torch.ones_like(V)
    dV_dS, = torch.autograd.grad(V, S, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)
    dV_dT, = torch.autograd.grad(V, T_vec, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)

    # 二阶偏导
    d2V_dS2, = torch.autograd.grad(dV_dS, S, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)

    # PDE残差（剩余期限T增大等价于时间t减小，故∂V/∂t = -∂V/∂T）
    pde_res = -dV_dT + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return pde_res

def train_pinn_pricer(df_train, df_val, STRIKE, T_MATURITY,
                       lambda_grid=[0.0, 0.01, 0.1, 1.0], seed=42):
    """
    训练PINN物理信息定价引擎（与BSM/Heston/SABR同级）
    输入：[S, K, T, r, σ] 输出：期权价格
    损失：数据拟合损失 + PDE物理约束损失
    超参选择：验证集RMSE最优的PDE权重λ
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info("训练PINN定价引擎 | 开始网格搜索PDE权重λ")

    def make_tensors(df):
        S = torch.tensor(df['S'].values, dtype=torch.float32)
        r = torch.tensor(df['r'].values, dtype=torch.float32)
        sigma = torch.tensor(df['target_vol_t1'].values, dtype=torch.float32)
        y = torch.tensor(df['true_bsm_price'].values, dtype=torch.float32)
        X = torch.stack([
            S, torch.full_like(S, STRIKE),
            torch.full_like(S, T_MATURITY), r, sigma
        ], dim=1)
        return X, y, S, sigma

    X_tr, y_tr, S_tr, sig_tr = make_tensors(df_train)
    X_val, y_val, _, _ = make_tensors(df_val)
    r_mean = float(df_train['r'].mean())

    best_lambda, best_val_rmse = 0.0, np.inf
    best_state_dict = None

    for lam in lambda_grid:
        net = PricingMLP(hidden_dims=PINN_HIDDEN_DIMS)
        optimizer = optim.Adam(net.parameters(), lr=PINN_LR)
        dataset = TensorDataset(X_tr, y_tr, S_tr, sig_tr)
        loader = DataLoader(dataset, batch_size=PINN_BATCH_SIZE, shuffle=False)

        net.train()
        for epoch in range(PINN_EPOCHS):
            for X_b, y_b, S_b, sig_b in loader:
                optimizer.zero_grad()
                pred = net(X_b)
                loss_data = nn.MSELoss()(pred, y_b)

                # PDE损失（λ=0时跳过，节省计算）
                if lam > 1e-8:
                    S_pde = torch.FloatTensor(len(S_b)).uniform_(float(S_b.min()), float(S_b.max()))
                    sig_pde = sig_b.detach().clone()
                    pde_res = bsm_pde_residual(net, S_pde, STRIKE, T_MATURITY, r_mean, sig_pde)
                    loss_pde = (pde_res ** 2).mean()
                    loss = loss_data + lam * loss_pde
                else:
                    loss = loss_data

                loss.backward()
                optimizer.step()

        # 验证集评估
        net.eval()
        with torch.no_grad():
            val_pred = net(X_val).numpy()
        val_rmse = np.sqrt(mean_squared_error(y_val.numpy(), val_pred))
        logger.info(f"  λ={lam:.3f} → 验证集RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_lambda = lam
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    logger.info(f"PINN最优λ={best_lambda} | 验证集最优RMSE={best_val_rmse:.4f}")

    # 加载最优模型，封装为向量化预测函数
    final_net = PricingMLP(hidden_dims=PINN_HIDDEN_DIMS)
    final_net.load_state_dict(best_state_dict)
    final_net.eval()

    def predict_price(S_arr, r_arr, sigma_arr):
        S_t = torch.tensor(S_arr, dtype=torch.float32)
        r_t = torch.tensor(r_arr, dtype=torch.float32)
        sig_t = torch.tensor(sigma_arr, dtype=torch.float32)
        X_t = torch.stack([
            S_t, torch.full_like(S_t, STRIKE),
            torch.full_like(S_t, T_MATURITY), r_t, sig_t
        ], dim=1)
        with torch.no_grad():
            pred = final_net(X_t).numpy()
        return np.maximum(pred, 0.0)

    meta = {'best_lambda': best_lambda, 'val_rmse': best_val_rmse, 'net': final_net}
    return predict_price, meta

# ===================== 模块3：分桶尾部风险评估 =====================
def run_tail_risk_evaluation(df_test, true_prices, model_price_dict,
                               baseline_prices, buckets=UNDERPRICE_BUCKETS,
                               bucket_labels=BUCKET_LABELS):
    """
    按真实低估幅度分桶评估误差 + 尾部风险指标统计
    核心逻辑：以基准价格为参照，看真实低估不同程度时，各模型的预测误差
    """
    logger.info("执行分桶尾部风险评估")

    # 计算真实价格相对基准的低估幅度，划分桶
    underprice_ratio = (baseline_prices - true_prices) / baseline_prices
    bucket_idx = np.digitize(underprice_ratio, buckets, right=True) - 1
    bucket_idx = np.clip(bucket_idx, 0, len(bucket_labels)-1)

    # ========== 1. 分桶RMSE统计 ==========
    bucket_records = []
    for i, label in enumerate(bucket_labels):
        mask = bucket_idx == i
        n_sample = mask.sum()
        if n_sample == 0:
            continue
        for model_name, pred_prices in model_price_dict.items():
            pred_valid = pred_prices[mask]
            true_valid = true_prices[mask]
            valid = np.isfinite(pred_valid) & np.isfinite(true_valid)
            if valid.sum() == 0:
                rmse, mae = np.nan, np.nan
            else:
                rmse = np.sqrt(mean_squared_error(true_valid[valid], pred_valid[valid]))
                mae = mean_absolute_error(true_valid[valid], pred_valid[valid])
            bucket_records.append({
                '低估区间': label, '样本量': int(n_sample),
                '模型': model_name, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)
            })

    bucket_df = pd.DataFrame(bucket_records)

    # ========== 2. 全样本尾部风险指标 ==========
    tail_records = []
    for model_name, pred_prices in model_price_dict.items():
        error = true_prices - pred_prices  # 正=模型低估，负=高估
        valid = np.isfinite(error)
        err_valid = error[valid]
        tail_records.append({
            '模型': model_name,
            '平均误差': round(np.mean(err_valid), 4),
            '误差中位数': round(np.median(err_valid), 4),
            '95分位低估幅度': round(np.percentile(err_valid, 95), 4),
            '99分位低估幅度': round(np.percentile(err_valid, 99), 4),
            '最大低估幅度': round(np.max(err_valid), 4),
            '低估样本占比': round((err_valid > 0).mean(), 4)
        })

    tail_df = pd.DataFrame(tail_records)

    # ========== 3. 可视化 ==========
    # 分桶RMSE柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=bucket_df, x='低估区间', y='RMSE', hue='模型', palette='viridis')
    plt.title('不同低估幅度下各模型定价RMSE对比', fontsize=13)
    plt.xlabel('真实价格相对基准的低估幅度', fontsize=11)
    plt.ylabel('RMSE', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/underprice_bucket_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 误差分布直方图
    plt.figure(figsize=(12, 6))
    for model_name, pred_prices in model_price_dict.items():
        error = true_prices - pred_prices
        sns.histplot(error, kde=True, label=model_name, alpha=0.5, bins=30)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='零误差线')
    plt.title('各模型定价误差分布（正数=模型低估价格）', fontsize=13)
    plt.xlabel('定价误差（真实价格 - 预测价格）', fontsize=11)
    plt.ylabel('样本频数', fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/pricing_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存结果
    bucket_df.to_csv('reports/underprice_bucket_rmse.csv', index=False)
    tail_df.to_csv('reports/tail_risk_metrics.csv', index=False)
    logger.info("分桶评估结果已保存至reports/目录")
    return bucket_df, tail_df

# ============================================================
# 主执行流程
# ============================================================
logger.info("=" * 60)
logger.info("启动新增模块：LSSM预测 + PINN定价 + 尾部风险评估")
logger.info("=" * 60)

# ---------- 1. 数据准备与数据集划分 ----------
# 补充真实BSM价格列（用当期真实波动率计算，作为定价基准）
df_final_with_price = df_final.copy()
df_final_with_price["true_bsm_price"] = black_scholes_vec(
    df_final_with_price["S"], STRIKE, T_MATURITY,
    df_final_with_price["r"], df_final_with_price["target_vol_t1"]
)

# 全局训练/测试集划分（与原有模型完全一致，保证对比公平）
df_train_full = df_final_with_price.iloc[:test_split_idx].copy()
df_test_all = df_final_with_price.iloc[test_split_idx:].copy()

# 从训练集中按时序切分验证集（后30%），严格无重叠
val_size = int(len(df_train_full) * 0.3)
val_size = max(val_size, 5)  # 边界保护，验证集至少5个样本
df_train_split = df_train_full.iloc[:-val_size].copy()
df_val_split = df_train_full.iloc[-val_size:].copy()

# 提取LSSM用的特征和标签
X_train_lssm = df_train_split[FEATURES]
y_train_lssm = df_train_split['target_vol_t1'].values
X_val_lssm = df_val_split[FEATURES]
y_val_lssm = df_val_split['target_vol_t1'].values

# ---------- 2. LSSM波动率预测 ----------
print("\n" + "=" * 60)
print("Step 1/3：LSSM波动率预测")
print("=" * 60)

lssm_pred_vol, lssm_meta = train_lssm_vol_predictor(
    X_train=X_train_lssm, y_train=y_train_lssm,
    X_val=X_val_lssm, y_val=y_val_lssm,
    X_test=X_test, features=FEATURES,
    n_components=LSSM_N_COMPONENTS,
    l2_transition=LSSM_L2_TRANSITION
)

# LSSM波动率效果评估
lssm_rmse, lssm_mae, lssm_r2, lssm_dir, _ = evaluate(y_test, lssm_pred_vol)
print(f"LSSM波动率预测 | RMSE={lssm_rmse:.4f} | R²={lssm_r2:.4f} | 方向准确率={lssm_dir:.2%}")

# ---------- 3. PINN定价引擎训练 ----------
print("\n" + "=" * 60)
print("Step 2/3：PINN物理信息定价引擎训练")
print("=" * 60)

pinn_predict, pinn_meta = train_pinn_pricer(
    df_train=df_train_split, df_val=df_val_split,
    STRIKE=STRIKE, T_MATURITY=T_MATURITY,
    lambda_grid=PINN_LAMBDA_GRID
)

# ---------- 4. 各组合定价计算 ----------
print("\n" + "=" * 60)
print("Step 3/3：全模型定价计算与评估")
print("=" * 60)

S_test = df_test['S'].values
r_test = df_test['r'].values
true_price_test = black_scholes_vec(S_test, STRIKE, T_MATURITY, r_test, y_test)
baseline_price_test = df_test['bsm_baseline'].values

# 复用原有ML最优波动率的定价结果
ml_bsm_price = pricing_output['bsm_price'].values
ml_heston_price = pricing_output['heston_price'].values
ml_sabr_price = pricing_output['sabr_price'].values
ml_ensemble_price = pricing_output['ensemble_price'].values
ml_pinn_price = pinn_predict(S_test, r_test, y_pred_vol)  # ML波动率+PINN定价
print(ml_bsm_price,ml_heston_price,ml_sabr_price,ml_ensemble_price,ml_pinn_price)

# LSSM波动率+各定价引擎
lssm_bsm_price = black_scholes_vec(S_test, STRIKE, T_MATURITY, r_test, lssm_pred_vol)
lssm_sabr_price = sabr_price_vec(S_test, STRIKE, T_MATURITY, r_test, lssm_pred_vol, **best_sabr_params)
lssm_pinn_price = pinn_predict(S_test, r_test, lssm_pred_vol)
# LSSM+Heston单独计算（仅一次，优化积分速度）
lssm_heston_price = heston_price_vec(S_test, STRIKE, T_MATURITY, r_test, lssm_pred_vol, **best_heston_params)
# LSSM集成定价
lssm_all_prices = np.array([lssm_bsm_price, lssm_heston_price, lssm_sabr_price, lssm_pinn_price])
lssm_ens_price = np.nanmean(lssm_all_prices, axis=0)

# ---------- 5. 统一尾部风险评估 ----------
all_model_prices = {
    # 基准模型
    "BSM基准(滞后波动率)": baseline_price_test,
    # ML最优波动率 + 各定价引擎
    "ML+BSM": ml_bsm_price,
    "ML+Heston": ml_heston_price,
    "ML+SABR": ml_sabr_price,
    "ML+PINN": ml_pinn_price,
    "ML+集成": ml_ensemble_price,
    # LSSM波动率 + 各定价引擎
    "LSSM+BSM": lssm_bsm_price,
    "LSSM+Heston": lssm_heston_price,
    "LSSM+SABR": lssm_sabr_price,
    "LSSM+PINN": lssm_pinn_price,
    "LSSM+集成": lssm_ens_price,
    # 原有端到端模型
    "E2E-XGBoost": e2e_pred
}

bucket_result, tail_result = run_tail_risk_evaluation(
    df_test=df_test,
    true_prices=true_price_test,
    model_price_dict=all_model_prices,
    baseline_prices=baseline_price_test
)

# ---------- 6. 结果打印 ----------
print("\n" + "=" * 70)
print("核心尾部风险指标（95分位=95%情况下低估不超过该值）")
print("=" * 70)
print(tail_result[['模型', '95分位低估幅度', '99分位低估幅度', '最大低估幅度', '平均误差']].to_string(index=False))

print("\n" + "=" * 70)
print("分低估区间RMSE对比")
print("=" * 70)
print(bucket_result.to_string(index=False))

# 保存全量定价结果
price_full = df_test[['date', 'regime']].copy()
price_full['真实价格'] = true_price_test
price_full['基准价格'] = baseline_price_test
for name, prices in all_model_prices.items():
    price_full[name] = prices
price_full.to_csv('reports/full_pricing_results_new.csv', index=False)

print("\n✅ 新增模块全部执行完成！结果已保存至reports/和plots/目录")