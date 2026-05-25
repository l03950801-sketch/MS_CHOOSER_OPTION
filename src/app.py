import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# 全局配置
# =========================================================
MS_COLORS = ["#1F77B4", "#FF6B6B", "#2E86AB", "#FFA500", "#00CC96"]
st.set_page_config(
    page_title="Volatility Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 专业CSS美化
# =========================================================
st.markdown("""
<style>
    .main { background-color: #0E1117; color: white; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    .metric-container {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    h1, h2, h3, h4, p, li { color: white !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 标题与说明
# =========================================================
st.title("波动率预测与期权定价分析平台")
st.markdown("""
本平台基于 **机器学习波动率预测 + BSM期权定价** 框架，支持：
- 波动率预测性能评估（Ridge / Random Forest / XGBoost）
- 分市场状态(Regime)模型对比
- 情绪因子敏感性分析
- SHAP模型可解释性分析
- 模型残差诊断与数据导出
""")

# =========================================================
# 数据加载（适配新版machine_learning.py输出）
# =========================================================
@st.cache_data(ttl=30)
def load_data():
    files = {
        # 核心波动率预测结果
        "vol_pred": "results/volatility_predictions.csv",
        # 模型整体性能表
        "performance": "results/model_performance.csv",
        # 分Regime性能结果
        "regime": "results/regime_analysis.csv",
        # 残差分析
        "residual": "results/residual_analysis.csv",
        # BSM希腊字母
        "greeks": "results/option_greeks.csv",
    }
    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            data[key] = df
        else:
            data[key] = pd.DataFrame()
    return data

data = load_data()
vol_df = data["vol_pred"]
perf_df = data["performance"]
regime_df = data["regime"]
residual_df = data["residual"]
greeks_df = data["greeks"]

# =========================================================
# 侧边栏 - 全局筛选器
# =========================================================
st.sidebar.title("控制面板")

# 日期筛选
min_date = pd.to_datetime("2018-01-01")
max_date = pd.to_datetime("2025-12-31")
date_range = st.sidebar.date_input("选择日期范围", [min_date, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)

# 市场状态筛选
regime_options = ["All Regimes", "Low_Vol", "High_Vol"]
selected_regime = st.sidebar.selectbox("选择市场状态", regime_options)

# 模型选择（适配新版：无E2E）
model_options = ["Linear Regression (Ridge L2)", "Random Forest", "XGBoost", "BSM Baseline (Lagged Vol)"]
selected_model = st.sidebar.selectbox("选择预测模型", model_options)

# =========================================================
# 全局数据过滤函数
# =========================================================
def filter_df(df):
    if df.empty or "date" not in df.columns:
        return df
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    if selected_regime != "All Regimes" and "regime" in df.columns:
        df = df[df["regime"] == selected_regime]
    return df

# 应用全局过滤
vol_df = filter_df(vol_df)
residual_df = filter_df(residual_df)
greeks_df = filter_df(greeks_df)

# =========================================================
# 模块化标签页（新增敏感性分析）
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "总览", "波动率预测", "市场状态",
    "敏感性分析", "SHAP解释性", "残差诊断", "数据导出"
])

# =========================================================
# TAB 1: 总览看板
# =========================================================
with tab1:
    st.header("模型总览")
    col1, col2, col3, col4 = st.columns(4)

    if not vol_df.empty and not perf_df.empty:
        # 提取核心指标
        rmse = perf_df[perf_df["Model"] == selected_model]["Volatility_RMSE"].values[0]
        r2 = perf_df[perf_df["Model"] == selected_model]["Volatility_R2"].values[0]
        acc = perf_df[perf_df["Model"] == selected_model]["Directional_Acc"].values[0]
        
        col1.metric("波动率 RMSE", f"{rmse:.4f}")
        col2.metric("拟合 R²", f"{r2:.4f}")
        col3.metric("方向准确率", f"{acc:.2%}")
        col4.metric("有效样本量", len(vol_df))
    else:
        for col in [col1, col2, col3, col4]:
            col.metric("N/A", "N/A")
        st.warning("无预测数据，请检查results文件路径")

    st.subheader("模型性能总表")
    if not perf_df.empty:
        st.dataframe(perf_df.round(4), use_container_width=True)
    else:
        st.info("模型性能数据未生成")

    st.success("""
    **核心框架**：机器学习波动率预测 + BSM期权定价 | 分市场状态分析 | 情绪敏感性测试
    """)

# =========================================================
# TAB 2: 波动率预测对比
# =========================================================
with tab2:
    st.header("波动率预测时序对比")
    if not vol_df.empty:
        fig = go.Figure()
        # 真实波动率
        fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["true_vol"], name="真实波动率", line=dict(color=MS_COLORS[0])))
        # 模型预测
        fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["ridge_pred"], name="Ridge预测", line=dict(color=MS_COLORS[1])))
        fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["rf_pred"], name="随机森林预测", line=dict(color=MS_COLORS[2])))
        fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["xgb_pred"], name="XGBoost预测", line=dict(color=MS_COLORS[3])))
        fig.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["bsm_pred"], name="BSM基准", line=dict(color=MS_COLORS[4])))

        fig.update_layout(template="plotly_dark", height=600, title="波动率预测对比")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("无波动率预测数据可展示")

    st.markdown("""
    ### 定价逻辑说明
    **Two-Step框架**：机器学习模型预测未来波动率 → 代入BSM公式计算期权价格
    **BSM基准**：使用滞后一期已实现波动率作为输入，作为行业基准对比
    """)

# =========================================================
# TAB 3: 市场状态(Regime)分析
# =========================================================
with tab3:
    st.header("分市场状态模型表现")
    if not regime_df.empty:
        st.dataframe(regime_df.round(4), use_container_width=True)
        fig = px.bar(regime_df, x="Regime", y="Volatility_RMSE", color="Model", 
                     barmode="group", template="plotly_dark", color_discrete_sequence=MS_COLORS)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("高波动市场下模型预测误差显著上升，符合波动率聚集特性")
    else:
        st.warning("无Regime分析数据")

# =========================================================
# TAB 4: 敏感性分析（核心新增！展示你的图表）
# =========================================================
with tab4:
    st.header("情绪因子敏感性分析")
    st.markdown("基于数据分布的扰动分析 | 低/高波动市场分离 | 95%经验置信带")
    
    col1, col2 = st.columns(2)
    with col1:
        low_vol_path = "plots/sensitivity_final_Low_Vol.png"
        if os.path.exists(low_vol_path):
            st.image(low_vol_path, caption="低波动市场敏感性分析", use_container_width=True)
        else:
            st.info("低波动市场敏感性图未生成")
    with col2:
        high_vol_path = "plots/sensitivity_final_High_Vol.png"
        if os.path.exists(high_vol_path):
            st.image(high_vol_path, caption="高波动市场敏感性分析", use_container_width=True)
        else:
            st.info("高波动市场敏感性图未生成")

# =========================================================
# TAB 5: SHAP可解释性AI
# =========================================================
with tab5:
    st.header("SHAP 模型可解释性")
    model_code_map = {
        "Linear Regression (Ridge L2)": "Ridge",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "BSM Baseline (Lagged Vol)": "BSM"
    }
    current_model = model_code_map[selected_model]

    # SHAP 蜜蜂图
    beeswarm_path = f"plots/shap_beeswarm_{current_model}.png"
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, use_container_width=True, caption="SHAP 蜜蜂图")
    else:
        st.info("该模型无SHAP蜜蜂图数据")

    # 分Regime特征重要性
    if selected_regime != "All Regimes":
        shap_path = f"plots/shap_{current_model}_{selected_regime}_importance.png"
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True, caption=f"{selected_regime} 特征重要性")

# =========================================================
# TAB 6: 残差诊断
# =========================================================
with tab6:
    st.header("波动率预测残差诊断")
    if not residual_df.empty:
        # 残差列匹配
        res_col_map = {
            "Linear Regression (Ridge L2)": "ridge_residual",
            "Random Forest": "rf_residual",
            "XGBoost": "xgb_residual",
            "BSM Baseline (Lagged Vol)": "bsm_residual"
        }
        res_col = res_col_map[selected_model]
        res_df = residual_df[["date", res_col]].dropna()
        res = res_df[res_col]

        # 残差统计指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("残差均值", f"{res.mean():.4f}")
        c2.metric("标准差", f"{res.std():.4f}")
        c3.metric("偏度", f"{stats.skew(res):.4f}")
        c4.metric("峰度", f"{stats.kurtosis(res):.4f}")

        # 残差分布
        fig_hist = px.histogram(res, nbins=30, template="plotly_dark", title="残差分布", color_discrete_sequence=[MS_COLORS[0]])
        st.plotly_chart(fig_hist, use_container_width=True)

        # 残差时序
        fig_ts = go.Figure(go.Scatter(x=res_df["date"], y=res_df[res_col], name="残差", line=dict(color=MS_COLORS[1])))
        fig_ts.update_layout(template="plotly_dark", title="残差时序变化", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)

    else:
        st.warning("无残差诊断数据")

# =========================================================
# TAB 7: 数据导出
# =========================================================
with tab7:
    st.header("数据导出")
    if not vol_df.empty:
        st.dataframe(vol_df.round(4), use_container_width=True)
        st.download_button(
            label="下载波动率预测数据(CSV)",
            data=vol_df.to_csv(index=False),
            file_name="volatility_prediction_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("无数据可导出")

# =========================================================
# 底部状态栏
# =========================================================
st.divider()
st.success(f"""
当前状态：模型 = {selected_model} | 市场状态 = {selected_regime} | 样本数量 = {len(vol_df)}
""")