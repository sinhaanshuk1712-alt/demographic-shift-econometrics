import streamlit as st
import wbgapi as wb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Demographic Shift & Econometrics",
    page_icon="📈",
    layout="wide"
)

# Available Countries and Regional Aggregates
COUNTRY_MAP = {
    "Japan": "JPN",
    "South Asia (Region)": "SAS",
    "India": "IND",
    "United States": "USA",
    "China": "CHN",
    "Germany": "DEU",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Sub-Saharan Africa (Region)": "SSF",
    "European Union": "EUU",
    "World": "WLD"
}

# --- Data Caching & Fetching ---
@st.cache_data(show_spinner=False)
def load_world_bank_data(country_codes):
    """
    Fetches Working-Age Population % (SP.POP.1564.TO.ZS) and 
    Constant GDP per Capita (NY.GDP.PCAP.KD) from 1990 to 2025.
    """
    time_range = range(1990, 2026)  # Includes 2025
    
    # 1. Fetch data from World Bank API
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', country_codes, time=time_range, numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', country_codes, time=time_range, numericTimeKeys=True)
    
    # 2. Reshape into long (tidy) format
    pop_tidy = df_pop.T.stack().reset_index()
    pop_tidy.columns = ['Year', 'Country_Code', 'Working_Age_Pct']
    
    gdp_tidy = df_gdp.T.stack().reset_index()
    gdp_tidy.columns = ['Year', 'Country_Code', 'GDP_Per_Capita']
    
    # 3. Merge indicators
    merged = pd.merge(pop_tidy, gdp_tidy, on=['Year', 'Country_Code'])
    merged['Year'] = pd.to_numeric(merged['Year'])
    merged['Log_GDP'] = np.log(merged['GDP_Per_Capita'])
    
    # Map country codes back to human-readable names
    code_to_name = {v: k for k, v in COUNTRY_MAP.items()}
    merged['Country'] = merged['Country_Code'].map(code_to_name)
    
    # Drop rows where either metric is unrecorded
    merged.dropna(subset=['Working_Age_Pct', 'GDP_Per_Capita'], inplace=True)
    
    return merged

# --- Sidebar Controls ---
st.sidebar.title("Configuration & Filters")

# 1. Country Selection
selected_country_names = st.sidebar.multiselect(
    "Select Countries / Regions:",
    options=list(COUNTRY_MAP.keys()),
    default=["Japan", "South Asia (Region)", "India", "United States"]
)

if not selected_country_names:
    st.warning("Please select at least one country or region from the sidebar.")
    st.stop()

selected_codes = [COUNTRY_MAP[name] for name in selected_country_names]

# 2. Year Range Slider
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=1990,
    max_value=2025,
    value=(1990, 2025),
    step=1
)

# Fetch Data
with st.spinner("Retrieving latest data from World Bank API..."):
    df_all = load_world_bank_data(tuple(selected_codes))

# Filter by selected year range
df_filtered = df_all[
    (df_all['Year'] >= year_range[0]) & 
    (df_all['Year'] <= year_range[1])
].sort_values(['Country', 'Year'])

# --- Main Dashboard ---
st.title("The Demographic Shift & Labor Interdependency")
st.markdown("#### Econometric Analysis of Working-Age Population and GDP Growth (1990–2025)")
st.write(
    "This dashboard tracks demographic transition dynamics using live World Bank data. "
    "Filter by region, adjust timeframes, and inspect OLS regression relationships below."
)

# Key Metrics Row (Latest Snapshot for first selected country)
latest_country = selected_country_names[0]
df_latest_c = df_filtered[df_filtered['Country'] == latest_country]

if not df_latest_c.empty:
    latest_row = df_latest_c.iloc[-1]
    earliest_row = df_latest_c.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Focus Country", latest_country)
    col2.metric(
        f"Working-Age Pop ({int(latest_row['Year'])})",
        f"{latest_row['Working_Age_Pct']:.2f}%",
        f"{(latest_row['Working_Age_Pct'] - earliest_row['Working_Age_Pct']):.2f}% since {int(earliest_row['Year'])}"
    )
    col3.metric(
        f"GDP per Capita ({int(latest_row['Year'])})",
        f"${latest_row['GDP_Per_Capita']:,.0f}",
        f"{((latest_row['GDP_Per_Capita'] - earliest_row['GDP_Per_Capita']) / earliest_row['GDP_Per_Capita'] * 100):.1f}%"
    )
    col4.metric("Available Timeframe", f"{int(earliest_row['Year'])} – {int(latest_row['Year'])}")

st.divider()

# --- Section 1: Interactive Charts ---
st.subheader("1. Macroeconomic & Demographic Trends")

tab1, tab2, tab3 = st.tabs([
    "📈 Working-Age Population (%)", 
    "💰 GDP per Capita (Constant 2015 US$)", 
    "🔍 Correlation & Trendlines"
])

with tab1:
    fig_pop = px.line(
        df_filtered,
        x="Year",
        y="Working_Age_Pct",
        color="Country",
        markers=True,
        title=f"Working-Age Population (% of Total, Ages 15-64) [{year_range[0]}–{year_range[1]}]",
        labels={"Working_Age_Pct": "Working-Age Population (% of Total)", "Year": "Year"}
    )
    fig_pop.update_layout(hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig_pop, use_container_width=True)

with tab2:
    scale_type = st.radio("Scale:", ["Linear", "Logarithmic"], horizontal=True, key="gdp_scale")
    y_col = "Log_GDP" if scale_type == "Logarithmic" else "GDP_Per_Capita"
    y_label = "Natural Log of GDP per Capita" if scale_type == "Logarithmic" else "GDP per Capita (Constant 2015 US$)"
    
    fig_gdp = px.line(
        df_filtered,
        x="Year",
        y=y_col,
        color="Country",
        markers=True,
        title=f"Economic Output per Capita [{year_range[0]}–{year_range[1]}]",
        labels={y_col: y_label, "Year": "Year"}
    )
    fig_gdp.update_layout(hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig_gdp, use_container_width=True)

with tab3:
    fig_scatter = px.scatter(
        df_filtered,
        x="Working_Age_Pct",
        y="Log_GDP",
        color="Country",
        trendline="ols",
        hover_data=["Year", "GDP_Per_Capita"],
        title="Working-Age Population Ratio vs. Log GDP per Capita",
        labels={
            "Working_Age_Pct": "Working-Age Population (% of Total)",
            "Log_GDP": "Log GDP per Capita"
        }
    )
    fig_scatter.update_layout(legend_title_text="")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Section 2: Econometric OLS Regression ---
st.subheader("2. Ordinary Least Squares (OLS) Econometric Estimation")
st.write("Model specification: $\\ln(\\text{GDP per Capita}_t) = \\beta_0 + \\beta_1 (\\text{Working-Age Pct}_t) + \\varepsilon_t$")

reg_country = st.selectbox("Select Country / Region to Estimate:", options=selected_country_names)

df_reg = df_filtered[df_filtered['Country'] == reg_country].dropna(subset=['Working_Age_Pct', 'Log_GDP'])

if len(df_reg) < 5:
    st.error("Insufficient data points in the selected timeframe to fit a regression model.")
else:
    X = sm.add_constant(df_reg['Working_Age_Pct'])
    y = df_reg['Log_GDP']
    model = sm.OLS(y, X).fit()
    
    # Statistical Summary Cards
    beta_1 = model.params['Working_Age_Pct']
    p_val = model.pvalues['Working_Age_Pct']
    r_sq = model.rsquared
    adj_r_sq = model.rsquared_adj
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coefficient (β₁)", f"{beta_1:.4f}")
    m2.metric("p-value", f"{p_val:.3e}")
    m3.metric("R² Score", f"{r_sq:.3f}")
    m4.metric("Adjusted R²", f"{adj_r_sq:.3f}")
    
    # Economic Interpretation
    pct_effect = (np.exp(beta_1) - 1) * 100
    st.info(
        f"**Economic Interpretation for {reg_country}:** "
        f"A 1 percentage point shift in the working-age population correlates with an estimated "
        f"**{pct_effect:+.2f}%** change in real GDP per capita ($p = {p_val:.3e}$). "
        f"The model explains **{r_sq * 100:.1f}%** of the observed variation in log GDP per capita."
    )
    
    # Detailed Statsmodels Printout
    with st.expander("Full Statistical Regression Table"):
        st.code(model.summary().as_text(), language="text")

# --- Section 3: Data Inspection & Export ---
with st.expander("Raw Data & Download"):
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv,
        file_name=f"world_bank_demographics_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )
