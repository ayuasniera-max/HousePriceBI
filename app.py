import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from ml_model import model

# 1. Page Configuration
st.set_page_config(
    page_title="House Price Intelligence",
    layout="wide"
)

# 2. Premium CSS Styling
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #f1f5f9; }

/* Header Styling */
.main-header {
    background: linear-gradient(135deg, #1e40af, #0891b2);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    color: white;
}

/* UI Containers */
.chart-box, .prediction-box {
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
}

/* KPI Card Refinement */
.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
}
.kpi-title { font-size: 14px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 5px; }

h1, h2, h3 { color: #0f172a; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# 3. Data Loading & Filtering
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_house.csv")

df = load_data()

# Navigation - Added ML Prediction directly to navigation to clean up layout
st.sidebar.title("🏡 House BI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview Dashboard", "Price Trends", "Neighborhood Insights", "Property Features", "ML Price Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

selected_area = st.sidebar.multiselect(
    "Select Neighborhoods",
    sorted(df["Neighborhood"].unique()),
    default=sorted(df["Neighborhood"].unique())[:5] # Defaulting to first 5 keeps charts clean initially
)

selected_price = st.sidebar.multiselect(
    "Select Price Categories",
    df["PriceCategory"].unique(),
    default=df["PriceCategory"].unique()
)

filtered_df = df[
    (df["Neighborhood"].isin(selected_area)) &
    (df["PriceCategory"].isin(selected_price))
]

# Universal Header
st.markdown(f"""
<div class="main-header">
    <h1 style="color:white; margin:0; font-size:28px;">House Price Intelligence</h1>
    <p style="margin:5px 0 0 0; opacity:0.85;">{page} | Market Trends & Property Valuation Insights</p>
</div>
""", unsafe_allow_html=True)

# 4. Metrics & Figures Pre-calculations
if not filtered_df.empty:
    avg_price = round(filtered_df["SalePrice"].mean(), 0)
    max_price = filtered_df["SalePrice"].max()
    total_house = filtered_df.shape[0]
    avg_age = round(filtered_df["HouseAge"].mean(), 1)

    # Cohesive Plotly Themes Using Your Brand Colors
    color_scale = ["#1e40af", #3b82f6, #60a5fa, #93c5fd, #cbd5e1]
    
    trend = filtered_df.groupby("YearBuilt")["SalePrice"].mean().reset_index()
    fig1 = px.line(trend, x="YearBuilt", y="SalePrice", title="Historical House Price Trend", template="plotly_white", color_discrete_sequence=["#1e40af"])
    
    bar = filtered_df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(bar, x="Neighborhood", y="SalePrice", title="Top 10 Neighborhoods by Average Price", template="plotly_white", color_discrete_sequence=["#3b82f6"])
    
    fig3 = px.pie(filtered_df, names="PriceCategory", title="Market Property Distribution", template="plotly_white", color_discrete_sequence=color_scale)
    
    fig4 = px.scatter(filtered_df, x="GrLivArea", y="SalePrice", color="PriceCategory", title="Living Area vs Sale Price", template="plotly_white", color_discrete_sequence=color_scale)
    
    fig5 = px.histogram(filtered_df, x="HouseStyle", color="PriceCategory", title="House Style Distribution by Price Class", template="plotly_white", color_discrete_sequence=color_scale)

# 5. Page Routing
if filtered_df.empty:
    st.warning("⚠️ No data available matching the selected filter criteria.")

elif page == "Overview Dashboard":
    # Clean KPI Row Layout
    k1, k2, k3, k4 = st.columns(4)
    metrics = [("Avg Sale Price", f"${avg_price:,.0f}"), ("Highest Price", f"${max_price:,.0f}"), ("Total Volume", f"{total_house:,}"), ("Avg House Age", f"{avg_age} Yrs")]
    for col, (title, val) in zip([k1, k2, k3, k4], metrics):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Just 2 clean, high-level structural charts on the main summary page
    row1_left, row1_right = st.columns([2, 1])
    with row1_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row1_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Price Trends":
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Neighborhood Insights":
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Property Features":
    row1, row2 = st.columns(2)
    with row1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.subheader("Feature Correlation Matrix")
        corr = filtered_df[["SalePrice", "GrLivArea", "GarageCars", "BedroomAbvGr", "HouseAge"]].corr()
        fig, ax = plt.subplots(figsize=(8, 5.2))
        sns.heatmap(corr, annot=True, ax=ax, cmap="Blues", cbar=False)
        fig.patch.set_facecolor('None') # Make matplotlib background transparent to fit clean card
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "ML Price Predictor":
    st.markdown("""
    <div class="prediction-box">
        <h3 style="margin-top:0;">Predictive Valuation Tool</h3>
        <p style="color:#64748b; margin-bottom:0;">Adjust the structural inputs below to generate an instant ML-driven property appraisal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    ml_left, ml_right = st.columns([2, 1])
    
    with ml_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Property Parameters</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            living = st.slider("Living Area (sq ft)", 500, 6000, 1500, step=100)
            bedroom = st.slider("Bedrooms", 1, 8, 3)
            age = st.slider("House Age (Years)", 0, 150, 20)
        with col2:
            garage = st.slider("Garage Capacity (Cars)", 0, 5, 2)
            quality = st.slider("Overall Material/Finish Quality", 1, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with ml_right:
        prediction = model.predict([[living, garage, bedroom, quality, age]])
        pred_val = prediction[0]
        
        st.markdown('<div class="chart-box" style="text-align: center; height:100%;">', unsafe_allow_html=True)
        st.metric(label="Estimated Market Value", value=f"${pred_val:,.0f}")
        
        if pred_val < 150000:
            st.success("🏷️ Affordable Tier")
        elif pred_val < 300000:
            st.info("📊 Mid-Range Tier")
        elif pred_val < 500000:
            st.warning("💎 High-Value Tier")
        else:
            st.error("👑 Luxury Tier")
        st.markdown('</div>', unsafe_allow_html=True)
