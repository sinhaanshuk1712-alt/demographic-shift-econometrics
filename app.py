import streamlit as st
import wbgapi as wb
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Macro-Demographic Panel Data Analysis")
st.subheader("Asian Tigers vs. Emerging South Asia (1990–2024)")

# Cache the API call so the app doesn't slow down on every refresh
@st.cache_data
def load_data():
    countries = ['KOR', 'SGP', 'IND', 'BGD']
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', countries, time=range(1990, 2025), numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', countries, time=range(1990, 2025), numericTimeKeys=True)
    
    # Reshape to Panel Data format
    df_pop_long = df_pop.reset_index().melt(id_vars='economy', var_name='Year', value_name='Working_Age_Pct')
    df_gdp_long = df_gdp.reset_index().melt(id_vars='economy', var_name='Year', value_name='GDP_Per_Capita')
    
    # Merge and clean
    df_panel = pd.merge(df_pop_long, df_gdp_long, on=['economy', 'Year'])
    df_panel.dropna(inplace=True)
    df_panel['Log_GDP'] = np.log(df_panel['GDP_Per_Capita'])
    return df_panel

st.write("Fetching live data from World Bank API...")
df_panel = load_data()

st.write("### Demographic Divergence: Working-Age Population Trends")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_panel, x='Year', y='Working_Age_Pct', hue='economy', ax=ax, linewidth=2.5)
ax.set_title("Population Ages 15-64 (% of Total)")
ax.set_ylabel("Working Age (%)")
st.pyplot(fig)

st.write("### Two-Way Fixed Effects Panel Regression")
st.markdown("This model controls for **Country Fixed Effects** (unobserved national traits) and **Time Fixed Effects** (global macroeconomic shocks) to prevent spurious correlations.")
st.code("Log_GDP ~ Working_Age_Pct + C(economy) + C(Year)")

# Fit and display the model
model = smf.ols('Log_GDP ~ Working_Age_Pct + C(economy) + C(Year)', data=df_panel).fit()
st.text(model.summary())
