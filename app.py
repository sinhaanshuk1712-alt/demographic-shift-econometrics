import streamlit as st
import wbgapi as wb
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Page Configuration
st.set_page_config(page_title="Demographic Shift & GDP Analysis", layout="wide")

st.title("The Demographic Shift & Labor Interdependence")
st.caption("An Econometric Analysis of Japan and South Asia (1990–2024) | Data Source: World Bank API (`wbgapi`)")

# Cache data retrieval for speed
@st.cache_data
def load_data():
    countries = ['JPN', 'SAS']
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', countries, time=range(1990, 2025), numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', countries, time=range(1990, 2025), numericTimeKeys=True)
    
    df_pop = df_pop.T.rename(columns={'JPN': 'Japan (Working Age %)', 'SAS': 'South Asia (Working Age %)'})
    df_gdp = df_gdp.T.rename(columns={'JPN': 'Japan GDP/Capita', 'SAS': 'South Asia GDP/Capita'})
    
    df_merged = pd.concat([df_pop, df_gdp], axis=1).dropna()
    df_merged['Japan_Log_GDP'] = np.log(df_merged['Japan GDP/Capita'])
    return df_merged

df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Filter the timeframe to analyze historical macroeconomic shifts.")
year_range = st.sidebar.slider("Select Year Range", int(df.index.min()), int(df.index.max()), (1990, 2023))

filtered_df = df.loc[year_range[0]:year_range[1]]

# --- KPI METRIC CARDS ---
start_year, end_year = year_range[0], year_range[1]
jpn_delta = filtered_df.loc[end_year, 'Japan (Working Age %)'] - filtered_df.loc[start_year, 'Japan (Working Age %)']
sas_delta = filtered_df.loc[end_year, 'South Asia (Working Age %)'] - filtered_df.loc[start_year, 'South Asia (Working Age %)']

col1, col2, col3 = st.columns(3)
col1.metric(
    label=f"Japan Working-Age Shift ({start_year}–{end_year})",
    value=f"{filtered_df.loc[end_year, 'Japan (Working Age %)']:.1f}%",
    delta=f"{jpn_delta:.1f}%",
    delta_color="normal"
)
col2.metric(
    label=f"South Asia Working-Age Shift ({start_year}–{end_year})",
    value=f"{filtered_df.loc[end_year, 'South Asia (Working Age %)']:.1f}%",
    delta=f"{sas_delta:.1f}%",
    delta_color="normal"
)
col3.metric(
    label="Japan Real GDP/Capita (Latest)",
    value=f"${filtered_df.loc[end_year, 'Japan GDP/Capita']:,.0f}",
    delta=f"{(filtered_df.loc[end_year, 'Japan GDP/Capita'] - filtered_df.loc[start_year, 'Japan GDP/Capita']):,.0f} USD"
)

st.markdown("---")

# --- SECTION 1: VISUALIZATIONS ---
tab1, tab2 = st.tabs(["Working-Age Population Share", "GDP per Capita (USD)"])

with tab1:
    st.subheader("1. The Demographic Divergence")
    st.line_chart(filtered_df[['Japan (Working Age %)', 'South Asia (Working Age %)']])

with tab2:
    st.subheader("2. Capital Divergence (Constant 2015 USD)")
    st.line_chart(filtered_df[['Japan GDP/Capita', 'South Asia GDP/Capita']])

st.markdown("---")

# --- SECTION 2: ECONOMETRIC REGRESSION ---
st.subheader("3. Econometric Proof: Labor Supply vs. Economic Output")
st.write("Ordinary Least Squares (OLS) specification: $\\ln(\\text{GDP}_t) = \\beta_0 + \\beta_1 (\\text{Working Age Share}_t) + \\epsilon_t$")

run_model = st.checkbox("Run Dynamic OLS Regression", value=True)

if run_model:
    X = sm.add_constant(filtered_df['Japan (Working Age %)'])
    y = filtered_df['Japan_Log_GDP']
    model = sm.OLS(y, X).fit()
    
    st.write(f"**Regression Output ({start_year} – {end_year}):**")
    st.text(model.summary())
    
    beta_val = model.params['Japan (Working Age %)']
    p_val = model.pvalues['Japan (Working Age %)']
    r2_val = model.rsquared
    
    st.info(
        f"**Economic Interpretation:** For every 1 percentage point decline in Japan's working-age population share, "
        f"real GDP per capita contracts by approximately **{beta_val * 100:.2f}%** (p-value: {p_val:.4e}, $R^2$: {r2_val:.3f})."
    )
