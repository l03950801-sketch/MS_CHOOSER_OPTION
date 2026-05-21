import streamlit as st
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化缓存
if "refresh_time" not in st.session_state:
    st.session_state.refresh_time = time.time()

# -------------------------- 统一工具函数（全局交互核心） --------------------------
def filter_data(df, start_date, end_date, regime):
    """统一数据过滤函数：日期 + Regime 全局联动"""
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & 
            (df["date"] <= pd.to_datetime(end_date))]
    if regime != "All Regimes":
        df = df[df["regime"] == regime]
    return df

# -------------------------- 侧边栏交互控件（核心筛选器） --------------------------
st.sidebar.title("控制面板")

# 1. 自动刷新交互（修复生效）
auto_refresh = st.sidebar.checkbox("自动刷新", value=False)
refresh_interval = st.sidebar.slider("刷新间隔(秒)", 5, 30, 10)
if auto_refresh and time.time() - st.session_state.refresh_time > refresh_interval:
    st.session_state.refresh_time = time.time()
    st.rerun()

# 2. 日期筛选（全局联动）
st.sidebar.subheader("日期筛选")
min_date, max_date = pd.to_datetime("2018-01-01"), pd.to_datetime("2025-12-31")
start_date, end_date = st.sidebar.date_input("选择时间范围", [min_date, max_date], min_value=min_date, max_value=max_date)

# 3. 波动状态筛选（全局联动）
st.sidebar.subheader("波动状态")
regime_options = ["All Regimes", "Low_Vol", "Normal_Vol", "High_Vol"]
selected_regime = st.sidebar.selectbox("选择Regime", regime_options, index=0)

# 4. 模型选择（全局联动）
st.sidebar.subheader("模型选择")
model_options = ["Linear Regression (Ridge)", "Random Forest", "XGBoost", "E2E Pricing"]
selected_model = st.sidebar.selectbox("选择模型", model_options, index=0)

# 模型编码映射
model_code_map = {
    "Linear Regression (Ridge)": "LR_Ridge",
    "Random Forest": "RandomForest",
    "XGBoost": "XGBoost",
    "E2E Pricing": "E2E"
}
current_model = model_code_map[selected_model]

# -------------------------- 加载数据 --------------------------
@st.cache_data(ttl=10)
def load_data():
    data = {}
    files = {
        "pricing": "reports/dual_pricing_with_95CI.csv",
        "residual": "reports/full_residual_analysis.csv",
        "regime": "reports/regime_all_models_results.csv",
        "greeks": "reports/option_greeks.csv",
        "governance": "reports/model_governance_report.csv"
    }
    for key, path in files.items():
        data[key] = pd.read_csv(path) if os.path.exists(path) else None
    return data

data = load_data()

# 全局统一过滤所有数据（交互核心）
pricing_df = filter_data(data["pricing"], start_date, end_date, selected_regime)
residual_df = filter_data(data["residual"], start_date, end_date, selected_regime)
greeks_df = filter_data(data["greeks"], start_date, end_date, selected_regime)
regime_df = data["regime"]

# -------------------------- 主页面 --------------------------
st.title("MS CHOOSER | 期权定价与波动率预测平台")
st.divider()

# -------------------------- 1. 核心指标（实时联动） --------------------------
st.subheader("核心指标")
cols = st.columns(4)
if not pricing_df.empty:
    pricing_df["in_ci"] = (pricing_df["bsm_baseline"] >= pricing_df["two_step_price_lower"]) & \
                          (pricing_df["bsm_baseline"] <= pricing_df["two_step_price_upper"])
    coverage = pricing_df["in_ci"].mean() * 100
    rmse = np.sqrt(mean_squared_error(pricing_df["bsm_baseline"], pricing_df["two_step_price"]))
    r2 = r2_score(pricing_df["bsm_baseline"], pricing_df["two_step_price"])
    
    cols[0].metric("95%置信覆盖率", f"{coverage:.2f}%")
    cols[1].metric("定价RMSE", f"{rmse:.4f}")
    cols[2].metric("拟合R²", f"{r2:.4f}")
else:
    cols[0].metric("覆盖率", "N/A")
    cols[1].metric("RMSE", "N/A")
    cols[2].metric("R²", "N/A")
cols[3].metric("有效样本数", len(residual_df))
st.divider()

# -------------------------- 2. 双重定价走势图（联动筛选） --------------------------
st.subheader("双重定价 & 95%置信区间")
if not pricing_df.empty:
    chart_data = pricing_df.set_index("date")[["bsm_baseline", "two_step_price", "e2e_price", 
                                               "two_step_price_lower", "two_step_price_upper"]]
    st.line_chart(chart_data, use_container_width=True)
else:
    st.warning("无匹配数据，请调整筛选条件")
st.divider()

# -------------------------- 3. 模型拟合图(Parity Plot)（联动模型） --------------------------
st.subheader("拟合效果 | Actual vs Predicted")
parity_path = "plots/model_performance_parity_all.png"
if os.path.exists(parity_path):
    st.image(parity_path, use_container_width=True, caption="全模型波动率拟合对比")
else:
    st.info("请运行machine_learning.py生成图表")
st.divider()

# -------------------------- 4. 分Regime模型性能 --------------------------
st.subheader("全模型Regime性能对比")
if regime_df is not None:
    st.dataframe(regime_df.round(4), use_container_width=True)
st.divider()

# -------------------------- 5. 敏感性分析 + Vega --------------------------
st.subheader("期权价格敏感性分析")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("plots/sensitivity_analysis.png"):
        st.image("plots/sensitivity_analysis.png", caption="综合敏感性")
with col2:
    if os.path.exists("plots/vol_sensitivity_vega.png"):
        st.image("plots/vol_sensitivity_vega.png", caption="Vega(波动率敏感性)")
st.divider()

# -------------------------- 6. 期权希腊字母（联动筛选） --------------------------
st.subheader("希腊字母")
if not greeks_df.empty:
    g1, g2 = st.columns(2)
    with g1:
        st.line_chart(greeks_df.set_index("date")["delta"], color="#FF6B6B", title="Delta")
    with g2:
        st.line_chart(greeks_df.set_index("date")["vega"], color="#2E86AB", title="Vega")
else:
    st.warning("无希腊字母数据")
st.divider()

# -------------------------- 7. SHAP分析（完美联动模型+Regime） --------------------------
st.subheader(f"SHAP分析 | {selected_model} | {selected_regime}")

# 仅树模型展示Beeswarm
if selected_model in ["Random Forest", "XGBoost"] and os.path.exists("plots/shap_beeswarm.png"):
    st.image("plots/shap_beeswarm.png", use_container_width=True, caption="全局SHAP蜜蜂图")
elif selected_model == "Linear Regression (Ridge)":
    st.info("线性模型不支持SHAP蜜蜂图")

# 分Regime特征重要性（严格联动）
shap_img = f"plots/shap_{current_model}_{selected_regime}_importance.png"
if selected_regime != "All Regimes" and os.path.exists(shap_img):
    st.image(shap_img, use_container_width=True, caption=f"{selected_regime}特征重要性")

# 特征交互热力图
interact_img = f"plots/shap_interaction_heatmap_{current_model}.png"
if os.path.exists(interact_img):
    st.image(interact_img, use_container_width=True, caption="特征交互热力图")
st.divider()

# -------------------------- 8. 残差分析（完美联动） --------------------------
st.subheader(f"残差诊断 | {selected_model}")
if not residual_df.empty:
    # 自动匹配模型残差
    res_col = "e2e_price_residual" if selected_model == "E2E Pricing" else "two_step_price_residual"
    res = residual_df[res_col].dropna()
    
    # 统计指标
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("残差均值", f"{res.mean():.4f}")
    c2.metric("标准差", f"{res.std():.4f}")
    c3.metric("偏度", f"{stats.skew(res):.4f}")
    c4.metric("峰度", f"{stats.kurtosis(res):.4f}")
    
    # 双图可视化
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,5))
    ax1.hist(res, bins=25, alpha=0.7, color="#1f77b4")
    ax1.axvline(0, c="r", ls="--")
    ax1.set_title("残差分布")
    
    ax2.plot(residual_df["date"], res, c="#ff6b6b")
    ax2.axhline(0, c="k", ls="--")
    ax2.set_title("残差时序")
    st.pyplot(fig)
else:
    st.warning("无残差数据")
st.divider()

# -------------------------- 9. 数据下载 --------------------------
st.subheader(" 数据导出")
if not pricing_df.empty:
    st.dataframe(pricing_df.round(4), use_container_width=True)
    st.download_button(" 下载筛选后数据", pricing_df.to_csv(index=False), 
                       "filtered_pricing.csv", "text/csv")

# -------------------------- 底部状态 --------------------------
st.success(f" 当前状态：模型={selected_model} | Regime={selected_regime} | 样本数={len(residual_df)}")