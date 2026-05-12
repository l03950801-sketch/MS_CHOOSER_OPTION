# Week 5 & 6 Volatility Forecasting & Forward Option Pricing
import pandas as pd
import numpy as np
import sys
import os
import random
import pickle
import json
import matplotlib
import scipy
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
GAP_DAYS = ROLLING_WINDOW

for directory in ["models", "plots", "cv_results", "metadata"]:
    os.makedirs(directory, exist_ok=True)

df = pd.read_csv("data/processed_data.csv")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

df["rolling_vol"] = df["return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
df["target_vol_t1"] = df["rolling_vol"].shift(-1)

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

df["synthetic_oracle_price"] = black_scholes_vec(
    df["S"], STRIKE, T_MATURITY, df["r"], df["rolling_vol"]
)
df = df.dropna().reset_index(drop=True)

def build_features(data):
    df = data.copy()
    df['sent_lag1'] = df['sentiment'].shift(1)
    df['S_lag1'] = df['S'].shift(1)
    df['r_lag1'] = df['r'].shift(1)
    df['rv_lag1'] = df['rolling_vol'].shift(1)
    df['sent_ma5'] = df['sentiment'].rolling(5).mean()
    return df.dropna()

df_final = build_features(df)
FEATURES = ['sent_lag1', 'S_lag1', 'r_lag1', 'rv_lag1', 'sent_ma5']

n_total = len(df_final)
test_split_idx = int(n_total * (1 - TEST_SIZE))

df_train = df_final.iloc[:test_split_idx].copy()
df_test = df_final.iloc[test_split_idx:].copy()

X_train, X_test = df_train[FEATURES], df_test[FEATURES]
y_train, y_test = df_train['target_vol_t1'], df_test['target_vol_t1']

try:
    tscv = TimeSeriesSplit(n_splits=5, gap=GAP_DAYS)
except TypeError:
    tscv = TimeSeriesSplit(n_splits=5)

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + EPS))) * 100
    r2 = r2_score(y_true, y_pred)
    
    dir_acc = 0.0
    if len(y_true) >= 2:
        true_diff = np.sign(np.clip(y_true[1:] - y_true[:-1], -DIR_THRESHOLD, DIR_THRESHOLD))
        pred_diff = np.sign(np.clip(y_pred[1:] - y_pred[:-1], -DIR_THRESHOLD, DIR_THRESHOLD))
        dir_acc = np.mean(true_diff == pred_diff)
        
    return mse, mae, rmse, mape, dir_acc, r2

print("T+1 Volatility Forecasting & Forward Option Pricing")
print("Naive Forecast Performance (Persistence Model)")
vol_naive = df_test['rv_lag1']
naive_metrics = evaluate(y_test, vol_naive)
print(f"MSE: {naive_metrics[0]:.6f} | MAE: {naive_metrics[1]:.6f} | MAPE: {naive_metrics[3]:.2f}% | Directional Accuracy: {naive_metrics[4]:.2%}")

print("\nBase Machine Learning Model Performance")
rf_base = RandomForestRegressor(random_state=SEED)
rf_base.fit(X_train, y_train)
rf_base_pred = rf_base.predict(X_test)

xgb_base = XGBRegressor(random_state=SEED)
xgb_base.fit(X_train, y_train)
xgb_base_pred = xgb_base.predict(X_test)

rf_base_metrics = evaluate(y_test, rf_base_pred)
xgb_base_metrics = evaluate(y_test, xgb_base_pred)

print(f"Random Forest | MSE: {rf_base_metrics[0]:.6f} | MAE: {rf_base_metrics[1]:.6f} | MAPE: {rf_base_metrics[3]:.2f}%")
print(f"XGBoost       | MSE: {xgb_base_metrics[0]:.6f} | MAE: {xgb_base_metrics[1]:.6f} | MAPE: {xgb_base_metrics[3]:.2f}%")

base_model_perf = {
    "Random Forest": rf_base_metrics[0],
    "XGBoost": xgb_base_metrics[0]
}
best_base_model = min(base_model_perf, key=base_model_perf.get)
best_base_vol_pred = rf_base_pred if best_base_model == "Random Forest" else xgb_base_pred
improvement = (naive_metrics[0] - min(base_model_perf.values())) / naive_metrics[0] * 100

print(f"\nOptimal Base Model: {best_base_model} | MSE Improvement vs Baseline: {improvement:.2f}%")

print("\nFeature Importance - Volatility Forecasting")
imp_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': rf_base.feature_importances_ if best_base_model == "Random Forest" else xgb_base.feature_importances_
}).sort_values('Importance', ascending=False)
print(imp_df)

print("\nTwo-Step Option Pricing Performance")
benchmark_prices = df_test['synthetic_oracle_price']
predicted_prices = black_scholes_vec(
    df_test["S"], STRIKE, T_MATURITY, df_test["r"], best_base_vol_pred
)
pricing_mse = mean_squared_error(benchmark_prices, predicted_prices)
print(f"Pricing MSE: {pricing_mse:.6f}")

print("\nEnd-to-End Option Pricing Model Performance")
y_e2e_train = df_train['synthetic_oracle_price']
y_e2e_test = df_test['synthetic_oracle_price']

lr_e2e = LinearRegression()
lr_e2e.fit(X_train, y_e2e_train)
lr_e2e_pred = lr_e2e.predict(X_test)

xgb_e2e = XGBRegressor(random_state=SEED)
xgb_e2e.fit(X_train, y_e2e_train)
xgb_e2e_pred = xgb_e2e.predict(X_test)

lr_e2e_mse = mean_squared_error(y_e2e_test, lr_e2e_pred)
xgb_e2e_mse = mean_squared_error(y_e2e_test, xgb_e2e_pred)

best_e2e_model = "XGBoost" if xgb_e2e_mse < lr_e2e_mse else "Linear Regression"
e2e_mse = min(lr_e2e_mse, xgb_e2e_mse)
print(f"Optimal E2E Model: {best_e2e_model} | Pricing MSE: {e2e_mse:.6f}")

lr_pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

rf_params = {
    "n_estimators": [200, 300], "max_depth": [3,5,7],
    "min_samples_split": [2,4], "min_samples_leaf": [1,3],
    "random_state": [SEED], "n_jobs": [1]
}
rf_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

xgb_params = {
    "n_estimators": [200,300], "max_depth": [2,3,4], "learning_rate": [0.01,0.05,0.1],
    "subsample": [0.7,0.8], "colsample_bytree": [0.6,0.7,0.8],
    "min_child_weight": [1,3,5], "reg_lambda": [1,5,10],
    "random_state": [SEED], "n_jobs": [1]
}
xgb_search = RandomizedSearchCV(XGBRegressor(), xgb_params, n_iter=15, cv=tscv, random_state=SEED, n_jobs=1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
rf_pred = best_rf.predict(X_test)
xgb_pred = best_xgb.predict(X_test)

S_t1_test = df_test["S"].values
r_t1_test = df_test["r"].values
true_price_t1 = df_test["synthetic_oracle_price"].values

def price_forward(S, r, sigma_hat):
    return black_scholes_vec(S, STRIKE, T_MATURITY, r, sigma_hat)

baseline_price = price_forward(S_t1_test, r_t1_test, vol_naive)
model_prices = [price_forward(S_t1_test, r_t1_test, pred) for pred in [lr_pred, rf_pred, xgb_pred]]

vol_performance = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost", "Persistence Baseline"],
    "MAE": [evaluate(y_test, lr_pred)[1], evaluate(y_test, rf_pred)[1], 
            evaluate(y_test, xgb_pred)[1], naive_metrics[1]],
    "RMSE": [evaluate(y_test, lr_pred)[2], evaluate(y_test, rf_pred)[2], 
             evaluate(y_test, xgb_pred)[2], naive_metrics[2]],
    "R2": [evaluate(y_test, lr_pred)[5], evaluate(y_test, rf_pred)[5], 
           evaluate(y_test, xgb_pred)[5], naive_metrics[5]]
})
vol_performance.to_csv("volatility_performance.csv", index=False)

model_rmse = {
    "Linear Regression": evaluate(y_test, lr_pred)[2],
    "Random Forest": evaluate(y_test, rf_pred)[2],
    "XGBoost": evaluate(y_test, xgb_pred)[2]
}
best_model_name = min(model_rmse, key=model_rmse.get)
best_predictor = lr_pipeline if best_model_name == "Linear Regression" else (best_rf if best_model_name == "Random Forest" else best_xgb)

explainer_model = best_predictor if best_model_name in ["Random Forest", "XGBoost"] else best_xgb
sample_size = min(500, len(X_train))
X_shap = X_train.sample(sample_size, random_state=SEED)
explainer = shap.TreeExplainer(explainer_model)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("plots/shap_importance.png", dpi=300)
plt.close()

shap.summary_plot(shap_values, X_shap, show=False)
plt.title("SHAP Value Distribution")
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", dpi=300)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(df_test["date"], y_test, label="Realized Volatility (T+1)")
plt.plot(df_test["date"], best_predictor.predict(X_test), label=f"Predicted Volatility - {best_model_name}")
plt.title("T+1 Volatility: Realized vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("plots/vol_prediction.png", dpi=300)
plt.close()

pickle.dump(best_predictor, open("models/best_model_final.pkl", "wb"))

import sklearn, xgboost
with open("requirements.txt", "w") as f:
    f.write(f"scikit-learn=={sklearn.__version__}\n")
    f.write(f"xgboost=={xgboost.__version__}\n")
    f.write(f"shap=={shap.__version__}\n")
    f.write(f"pandas=={pd.__version__}\n")
    f.write(f"numpy=={np.__version__}\n")
    f.write(f"matplotlib=={matplotlib.__version__}\n")
    f.write(f"scipy=={scipy.__version__}\n")

metadata = {
    "features": FEATURES,
    "best_model": best_model_name,
    "volatility_annualization": TRADING_DAYS,
    "train_period": [str(df_train["date"].iloc[0]), str(df_train["date"].iloc[-1])],
    "test_period": [str(df_test["date"].iloc[0]), str(df_test["date"].iloc[-1])]
}
with open("metadata/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("\nAnalysis Completed Successfully")