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
    page_title="MS CHOOSER | Quant Dashboard",
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
st.title("MS CHOOSER | Explainable AI Option Pricing Platform")
st.markdown("""
本平台集成 **传统期权定价 + 机器学习波动率预测 + 可解释性**，支持：
- 波动率预测与期权双路径定价
- 市场状态(Regime)诊断
- SHAP模型可解释性分析
- 期权希腊字母(Greeks)风险分析
- 模型残差诊断与数据导出
""")

# =========================================================
# 数据加载（带缓存+过期机制）
# =========================================================
@st.cache_data(ttl=30)
def load_data():
    files = {
        "pricing": "reports/dual_pricing_with_95CI.csv",
        "residual": "reports/full_residual_analysis.csv",
        "regime": "reports/regime_all_models_results.csv",
        "greeks": "reports/option_greeks.csv",
        "governance": "reports/model_governance_report.csv"
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
pricing_df = data["pricing"]
residual_df = data["residual"]
regime_df = data["regime"]
greeks_df = data["greeks"]
gov_df = data["governance"]

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
regime_options = ["All Regimes", "Low_Vol", "Normal_Vol", "High_Vol"]
selected_regime = st.sidebar.selectbox("选择市场状态", regime_options)

# 模型选择
model_options = ["Linear Regression", "Random Forest", "XGBoost", "E2E Pricing"]
selected_model = st.sidebar.selectbox("选择机器学习模型", model_options)

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
pricing_df = filter_df(pricing_df)
residual_df = filter_df(residual_df)
greeks_df = filter_df(greeks_df)

# =========================================================
# 模块化标签页
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Pricing", "Regime",
    "SHAP", "Greeks", "Residual", "Export"
])

# =========================================================
# TAB 1: 总览看板
# =========================================================
with tab1:
    st.header("项目总览")
    col1, col2, col3, col4 = st.columns(4)

    if not pricing_df.empty:
        pricing_df["in_ci"] = (
            (pricing_df["bsm_baseline"] >= pricing_df["two_step_price_lower"]) &
            (pricing_df["bsm_baseline"] <= pricing_df["two_step_price_upper"])
        )
        coverage = pricing_df["in_ci"].mean() * 100
        rmse = np.sqrt(mean_squared_error(pricing_df["bsm_baseline"], pricing_df["two_step_price"]))
        r2 = r2_score(pricing_df["bsm_baseline"], pricing_df["two_step_price"])

        col1.metric("95% CI 覆盖率", f"{coverage:.2f}%")
        col2.metric("定价 RMSE", f"{rmse:.4f}")
        col3.metric("拟合 R²", f"{r2:.4f}")
        col4.metric("有效样本量", len(pricing_df))
    else:
        for col in [col1, col2, col3, col4]:
            col.metric("N/A", "N/A")
        st.warning("无定价数据，请检查文件路径")

    st.subheader("模型治理报告")
    if not gov_df.empty:
        st.dataframe(gov_df.round(4), use_container_width=True)
    else:
        st.info("模型治理数据未生成")

    st.success("""
    **核心框架**：Two-Step ML + BSM 定价 | 时间序列交叉验证 | 分市场状态诊断
    """)

# =========================================================
# TAB 2: 期权定价分析
# =========================================================
with tab2:
    st.header("期权定价对比")
    if not pricing_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pricing_df["date"], y=pricing_df["bsm_baseline"], name="BSM 基准", line=dict(color=MS_COLORS[0])))
        fig.add_trace(go.Scatter(x=pricing_df["date"], y=pricing_df["two_step_price"], name="两步法定价", line=dict(color=MS_COLORS[1])))
        fig.add_trace(go.Scatter(x=pricing_df["date"], y=pricing_df["e2e_price"], name="端到端定价", line=dict(color=MS_COLORS[2])))
        fig.add_trace(go.Scatter(x=pricing_df["date"], y=pricing_df["two_step_price_upper"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=pricing_df["date"], y=pricing_df["two_step_price_lower"], fill='tonexty', line=dict(width=0), name='95% 置信区间'))

        fig.update_layout(template="plotly_dark", height=600, title="定价趋势对比")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("无定价数据可展示")

    st.markdown("""
    ### 定价逻辑说明
    **Two-Step**：机器学习预测波动率 → BSM 模型定价（保留金融理论结构）
    **E2E**：直接拟合价格（无金融约束，易出现不合理定价）
    """)

# =========================================================
# TAB 3: 市场状态(Regime)分析
# =========================================================
with tab3:
    st.header("分市场状态模型表现")
    if not regime_df.empty:
        st.dataframe(regime_df.round(4), use_container_width=True)
        fig = px.bar(regime_df, x="Regime", y="RMSE", color="Model", barmode="group", template="plotly_dark", color_discrete_sequence=MS_COLORS)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("高波动市场下所有模型误差显著上升，验证BSM模型在极端行情下的局限性")
    else:
        st.warning("无Regime分析数据")

# =========================================================
# TAB 4: SHAP可解释性AI
# =========================================================
with tab4:
    st.header("SHAP 模型可解释性")
    model_code_map = {
        "Linear Regression": "LR_Ridge",
        "Random Forest": "RandomForest",
        "XGBoost": "XGBoost",
        "E2E Pricing": "E2E"
    }
    current_model = model_code_map[selected_model]

    # SHAP 蜜蜂图
    beeswarm_path = f"plots/shap_beeswarm_{current_model}.png"
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, use_container_width=True, caption="SHAP 蜜蜂图")
    else:
        st.info("ℹ️ 该模型无SHAP蜜蜂图数据")

    # 分Regime特征重要性
    if selected_regime != "All Regimes":
        shap_path = f"plots/shap_{current_model}_{selected_regime}_importance.png"
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True, caption=f"{selected_regime} 特征重要性")

    # 特征交互热力图
    interact_path = f"plots/shap_interaction_heatmap_{current_model}.png"
    if os.path.exists(interact_path):
        st.image(interact_path, use_container_width=True, caption="特征交互热力图")

# =========================================================
# TAB 5: 期权希腊字母(Greeks)
# =========================================================
with tab5:
    st.header("期权风险希腊字母")
    if not greeks_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_delta = px.line(greeks_df, x="date", y="delta", template="plotly_dark", title="Delta (价格敏感度)", color_discrete_sequence=[MS_COLORS[1]])
            st.plotly_chart(fig_delta, use_container_width=True)
        with col2:
            fig_vega = px.line(greeks_df, x="date", y="vega", template="plotly_dark", title="Vega (波动率敏感度)", color_discrete_sequence=[MS_COLORS[2]])
            st.plotly_chart(fig_vega, use_container_width=True)
    else:
        st.warning("无希腊字母数据")

# =========================================================
# TAB 6: 残差诊断（纯Plotly，无Matplotlib）
# =========================================================
with tab6:
    st.header("模型残差诊断")
    if not residual_df.empty:
        res_col = "e2e_price_residual" if selected_model == "E2E Pricing" else "two_step_price_residual"
        res_df = residual_df[["date", res_col]].dropna()
        res = res_df[res_col]

        # 残差统计指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("残差均值", f"{res.mean():.4f}")
        c2.metric("标准差", f"{res.std():.4f}")
        c3.metric("偏度", f"{stats.skew(res):.4f}")
        c4.metric("峰度", f"{stats.kurtosis(res):.4f}")

        # 残差分布直方图
        fig_hist = px.histogram(res, nbins=30, template="plotly_dark", title="残差分布", color_discrete_sequence=[MS_COLORS[0]])
        st.plotly_chart(fig_hist, use_container_width=True)

        # 残差时序图
        fig_ts = go.Figure(go.Scatter(x=res_df["date"], y=res_df[res_col], name="残差", line=dict(color=MS_COLORS[1])))
        fig_ts.update_layout(template="plotly_dark", title="残差时序变化", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)

        # QQ 图 (Plotly版本)
        st.subheader("正态性检验 QQ 图")
        qq = stats.probplot(res, dist="norm")
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="样本分位数", marker=dict(color=MS_COLORS[1])))
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode="lines", name="标准正态", line=dict(color=MS_COLORS[0])))
        qq_fig.update_layout(template="plotly_dark", title="QQ Plot", height=500)
        st.plotly_chart(qq_fig, use_container_width=True)
    else:
        st.warning("无残差诊断数据")

# =========================================================
# TAB 7: 数据导出
# =========================================================
with tab7:
    st.header("数据导出")
    if not pricing_df.empty:
        st.dataframe(pricing_df.round(4), use_container_width=True)
        st.download_button(
            label="下载定价数据(CSV)",
            data=pricing_df.to_csv(index=False),
            file_name="MS_Chooser_Option_Pricing.csv",
            mime="text/csv"
        )
    else:
        st.warning("无数据可导出")

# =========================================================
# 底部状态栏
# =========================================================
st.divider()
st.success(f"""
当前状态：模型 = {selected_model} | 市场状态 = {selected_regime} | 样本数量 = {len(pricing_df)}
""")