import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Intelligence Dashboard",
    layout="wide"
)

df = pd.read_csv("data/cleaned_house.csv")

st.markdown("""
<style>
.stApp { background-color: #eef2f7; }

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #101828;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-header {
    background: linear-gradient(90deg, #1d4ed8, #06b6d4);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    color: white;
}

.nav-box {
    background: white;
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.10);
    border-left: 6px solid #2563eb;
}

.kpi-title {
    font-size: 15px;
    color: #64748b;
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
}

.chart-box {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

h1, h2, h3 { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("House BI")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "",
    ["Dashboard", "Price Analysis", "Neighborhood", "Property Type", "Correlation Analysis"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

selected_area = st.sidebar.multiselect(
    "Neighborhood",
    sorted(df["Neighborhood"].unique()),
    default=sorted(df["Neighborhood"].unique())
)

selected_price = st.sidebar.multiselect(
    "Price Category",
    df["PriceCategory"].unique(),
    default=df["PriceCategory"].unique()
)

filtered_df = df[
    (df["Neighborhood"].isin(selected_area)) &
    (df["PriceCategory"].isin(selected_price))
]

st.markdown("""
<div class="main-header">
    <h1 style="color:white;">House Price Intelligence Dashboard</h1>
    <p>Business Intelligence dashboard for property price trends, neighborhood comparison, and housing market insights.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="nav-box">
    <b>Current Section:</b> {page}
</div>
""", unsafe_allow_html=True)

avg_price = round(filtered_df["SalePrice"].mean(), 0)
max_price = filtered_df["SalePrice"].max()
total_house = filtered_df.shape[0]
avg_age = round(filtered_df["HouseAge"].mean(), 1)

trend = filtered_df.groupby("YearBuilt")["SalePrice"].mean().reset_index()
fig1 = px.line(trend, x="YearBuilt", y="SalePrice", title="House Price Trend")

bar = filtered_df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False).head(10).reset_index()
fig2 = px.bar(bar, x="Neighborhood", y="SalePrice", title="Top 10 Neighborhoods by Average Price")

fig3 = px.pie(filtered_df, names="PriceCategory", title="Property Distribution")

fig4 = px.scatter(
    filtered_df,
    x="GrLivArea",
    y="SalePrice",
    color="Neighborhood",
    title="Living Area vs Sale Price"
)

fig5 = px.histogram(
    filtered_df,
    x="HouseStyle",
    color="PriceCategory",
    title="House Style Distribution"
)

if page == "Dashboard":
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Price</div>
            <div class="kpi-value">${avg_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Highest Price</div>
            <div class="kpi-value">${max_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Houses</div>
            <div class="kpi-value">{total_house}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average House Age</div>
            <div class="kpi-value">{avg_age}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row1_left, row1_right = st.columns([2, 1])

    with row1_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row3_left, row3_right = st.columns(2)

    with row3_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row3_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.subheader("Correlation Heatmap")

        corr = filtered_df[
            ["SalePrice", "GrLivArea", "GarageCars", "BedroomAbvGr", "HouseAge"]
        ].corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, ax=ax, cmap="Blues")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Price Analysis":
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Neighborhood":
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Property Type":
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Correlation Analysis":
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")

    corr = filtered_df[
        ["SalePrice", "GrLivArea", "GarageCars", "BedroomAbvGr", "HouseAge"]
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, ax=ax, cmap="Blues")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
