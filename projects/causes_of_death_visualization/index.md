# Tracking the Leading Causes of Death in the U.S. (1999–2017)

## Overview
This project uses public data from the National Center for Health Statistics (NCHS) to visualize long-term mortality trends in the United States from 1999 to 2017. The analysis focuses on national trends over time, geographic differences by state, and comparisons across major causes of death.

## Focus Causes
- Heart Disease
- Cancer
- Stroke
- Chronic Lower Respiratory Disease (CLRD)
- Unintentional Injuries

## Data
Source: NCHS (data.gov)  
Key fields used: year, state, cause of death, total deaths, and age-adjusted death rate.

## Visualizations & Insights
- Top causes by total deaths across the full period (Heart Disease and Cancer dominate)
- Heatmap of age-adjusted death rates showing declines in several causes over time while Unintentional Injuries rises
- Choropleth maps comparing Heart Disease death rates by state in 1999 vs. 2017 using a consistent scale
- Trend comparison showing why age-adjusted death rate can fall while total deaths increase (population growth + aging)
- Percent-change comparisons highlighting the sharp rise in Unintentional Injuries

## Tools & Technologies
- Python (Pandas, Matplotlib)
- R (usmap, ggplot2, dplyr)

## Code & Analysis
The full analysis workflow—including cleaning, aggregation, and visualization—was implemented in Python and R. The notebook reproduces the figures and summary comparisons used in the final report, and the R script generates the state-level choropleth maps.

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)
- 🧪 **Python Notebook:** [analysis.ipynb](./analysis.ipynb)
- 🗺️ **R Choropleth Code:** [choropleth_r_code.txt](./choropleth_r_code.txt)
