import streamlit as st
import wbgapi as wb
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page Configuration
st.set_page_config(page_title="Demographic Shift & GDP Analysis", layout="wide")

st.title("The Demographic Shift & Labor Interdependence")
st.write("An Econometric Analysis of Japan and South Asia (1990–2024)")
st.write("Built to analyze structural economic stagnation and bilateral labor integration.")

# Cache the API pull so the app loads instantly for admissions officers
@st.cache_data
def load_data():
    countries = ['JPN', 'SAS']
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', countries, time=range(1990, 2025), numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', countries, time=range(1990, 2025), numericTimeKeys=True)
    
    df_pop = df_pop.T.rename(columns={'JPN': 'JPN_Working_Age_Pct', 'SAS': 'SAS_Working_Age_Pct'})
    df_gdp = df_gdp.T.rename(columns={'JPN': 'JPN_GDP_Per_Capita', 'SAS': 'SAS_GDP_Per_Capita'})
    
    df_merged = pd.concat([df_pop, df_gdp], axis=1).dropna()
    df_merged['JPN_Log_GDP'] = np.log(df_merged['JPN_GDP_Per_Capita'])
    return df_merged

df = load_data()

# Visualization Section
st.subheader("1. The Demographic Divergence")
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_theme(style="whitegrid")
sns.lineplot(data=df, x=df.index, y='JPN_Working_Age_Pct', label='Japan', linewidth=2.5, ax=ax)
sns.lineplot(data=df, x=df.index, y='SAS_Working_Age_Pct', label='South Asia', linewidth=2.5, ax=ax)
ax.set_ylabel("Working-Age Population (% of Total)")
ax.set_xlabel("Year")
st.pyplot(fig)

# Regression Section
st.subheader("2. Econometric Proof: Demographics vs. GDP Growth in Japan")
st.write("This Ordinary Least Squares (OLS) regression models how the decline in Japan's working-age population correlates with its economic output (Log GDP).")

X = sm.add_constant(df['JPN_Working_Age_Pct'])
y = df['JPN_Log_GDP']
model = sm.OLS(y, X).fit()

# Display regression results cleanly
st.text(model.summary())
