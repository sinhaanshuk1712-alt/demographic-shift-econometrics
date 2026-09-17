# Demographic Shift & Labor Interdependence: An Econometric Analysis (Japan vs. South Asia, 1990–2024)
> **Live Interactive Dashboard:** [View Live Streamlit App](https://demographic-shift-econometrics-bzpazsji3eaogpgmvadzxc.streamlit.app/)
## Project Overview
This repository contains an empirical econometric analysis investigating the correlation between shifting working-age population shares and real GDP per capita growth across Japan and South Asia over a 34-year horizon. 

Using the official World Bank API (`wbgapi`) and Python econometric libraries, this study tests the hypothesis that Japan’s economic stagnation is heavily structural/demographic, highlighting the potential for bilateral labor integration with labor-surplus regions like South Asia.

---

## Indicators Analyzed (World Bank Database)
- **`SP.POP.1564.TO.ZS`**: Population ages 15–64 (% of total population)
- **`NY.GDP.PCAP.KD`**: GDP per capita (constant 2015 USD)

---

## Methodology
1. **Extraction & Alignment:** Time-series retrieval using Python's `wbgapi` for regions `JPN` (Japan) and `SAS` (South Asia) covering 1990–2024.
2. **Transformations:** Log transformation of GDP per capita to measure percentage growth sensitivity.
3. **Econometric Model:** Ordinary Least Squares (OLS) regression:
   $$\ln(\text{GDP}_t) = \beta_0 + \beta_1 (\text{Working Age Share}_t) + \epsilon_t$$

---

## Key Findings
- **Demographic Divergence:** Japan's working-age population peaked in the mid-1990s at ~69.8% and has steadily dropped below 59%, while South Asia's working-age share rose from ~56% to over 66% during the same window.
- **Econometric Significance:** The OLS model shows a statistically significant positive relationship ($\beta_1 > 0$, $p < 0.001$) between working-age demographic share and real output per capita in Japan, supporting the premise that demographic contraction acts as a primary drag on domestic output.

---

## Tech Stack
- **Data Retrieval:** `wbgapi`
- **Data Manipulation:** `pandas`, `numpy`
- **Econometric Modeling:** `statsmodels`
- **Visualization:** `matplotlib`, `seaborn`
