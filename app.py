import streamlit as st
import wbgapi as wb
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Page Configuration
st.set_page_config(page_title="Demographic Shift & GDP Analysis", layout="wide")

st.title("The Demographic Shift & Labor Interdependence")
st.write("An Econometric Analysis of Japan and South Asia (1990–2024)")

# Cache the API pull so the app loads instantly
@st.cache_data
def load_data():
    countries = ['JPN', 'SAS']
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', countries, time=range(1990, 2025), numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', countries, time=range(1990, 2025), numericTimeKeys=True)
    
    df_pop = df_pop.T.rename(columns={'JPN': 'Japan (Working Age %)', 'SAS': 'South Asia (Working Age %)'})
    df_gdp = df_gdp.T.rename(columns={'JPN': 'Japan_GDP', 'SAS': 'South Asia_GDP'})
    
    df_merged = pd.concat([df_pop, df_gdp], axis=1).dropna()
    df_merged['Japan_Log_GDP'] = np.log(df_merged['Japan_GDP'])
    return df_merged

df = load_data()

# --- INTERACTIVE DASHBOARD CONTROLS ---
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Use this slider to adjust the econometric timeframe.")
year_range = st.sidebar.slider("Select Year Range", int(df.index.min()), int(df.index.max()), (1990, 2023))

# Filter data based on user's slider input
filtered_df = df.loc[year_range[0]:year_range[1]]

# --- 1. VISUALIZATION SECTION ---
st.subheader("1. The Demographic Divergence (Interactive)")
st.write("Hover over the lines to view exact working-age population percentages for specific years.")
# Streamlit's native line_chart is interactive by default
st.line_chart(filtered_df[['Japan (Working Age %)', 'South Asia (Working Age %)']])

# --- 2. REGRESSION SECTION ---
st.subheader("2. Econometric Proof: Demographics vs. GDP Growth")
st.write("Click the checkbox below to run a live regression model based on your selected timeframe.")

# Interactive toggle for the math model
run_model = st.checkbox("Run OLS Regression on Selected Timeframe")

if run_model:
    X = sm.add_constant(filtered_df['Japan (Working Age %)'])
    y = filtered_df['Japan_Log_GDP']
    model = sm.OLS(y, X).fit()
    st.write(f"**Regression Results ({year_range[0]} - {year_range[1]}):**")
    st.text(model.summary())
