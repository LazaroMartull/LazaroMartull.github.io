# Analyzing Circadian Rhythms Through Twitter Activity

## Overview
This project analyzes global activity patterns using Twitter timestamps as a proxy for daily sleep/activity rhythms. We explored how **time zone/location** and **occupation** relate to shifts in posting behavior.

## Data
Dataset: **jobs_sleepwalk (2020)** with **4.5M+ rows** and key fields:
- characteristic (occupation / role)
- utc_timestamp (time)
- user_hash (anonymized user)
- location

## Methodology
- Converted UTC timestamps into datetime features for hourly analysis
- Cleaned location values
- Visualized trends by occupation and location (bar charts, hourly trends, heatmap)

## Tools
Python, Pandas, Matplotlib/Seaborn/Plotly

## Files
- 📄 Report: `report.pdf`
- 🧪 Notebook: `Project_Circadian_Rhythms.ipynb` (optional)
