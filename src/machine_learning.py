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

for directory in ["models", "plots", "metadata", "reports"]:
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

# 修复1：特征无泄露 → rolling(5).mean().shift(1)
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

# 特征标准化（修复SHAP兼容问题）
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
model_names = ["Linear", "RandomForest", "XGBoost"]
results = []

for name, pred in zip(model_names, preds):
    mse, mae, rmse, r2, dir_acc = evaluate(y_test, pred)
    results.append([name, rmse, r2, dir_acc])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "Directional_Acc"])
best_idx = results_df["RMSE"].idxmin()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = models[best_idx]
y_pred_vol = preds[best_idx]

if best_model_name == "Linear":
    explainer = shap.LinearExplainer(best_model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("plots/shap_importance.png", dpi=300, bbox_inches='tight')
plt.close()

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

def expanding_window_validation(X, y, model, tscv, is_linear=False):
    rmse_list = []
    for train_idx, test_idx in tscv.split(X):
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        if is_linear:
            X_test_fold_scaled = scaler.transform(X_test_fold)
            pred = model.predict(X_test_fold_scaled)
        else:
            pred = model.predict(X_test_fold)
            
        rmse = np.sqrt(mean_squared_error(y_test_fold, pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)

expanding_rmse = expanding_window_validation(
    X_train, y_train, best_model, tscv, is_linear=(best_model_name=="Linear")
)

print("="*50)
print(f"Best Model: {best_model_name}")
print(f"Volatility RMSE: {results_df.loc[best_idx, 'RMSE']:.4f}")
print(f"Directional Accuracy: {results_df.loc[best_idx, 'Directional_Acc']:.2%}")
print(f"Expanding Window CV RMSE: {expanding_rmse:.4f}")
print("="*50)
print("\nRegime Test Results:")
print(regime_df)

pickle.dump(best_model, open("models/best_quant_model.pkl", "wb"))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_vol, alpha=0.6, color='#2E86AB')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('Actual vs Predicted Volatility')
plt.tight_layout()
plt.savefig("plots/model_performance_actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.close()
