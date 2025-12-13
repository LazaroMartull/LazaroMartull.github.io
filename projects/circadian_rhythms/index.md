# Circadian Rhythm Analysis Using Twitter Activity

## Overview
This project analyzes global activity patterns using Twitter timestamps as a proxy for daily sleep–wake rhythms. The objective was to examine how **geographic location (time zone)** and **occupation** relate to shifts in online activity that may indicate circadian rhythm disruption.

## Data
**Dataset:** jobs_sleepwalk (2020)  
**Scale:** 4.5M+ records

Key fields:
- `characteristic` — occupation / role
- `utc_timestamp` — posting time (UTC)
- `user_hash` — anonymized user identifier
- `location` — user-reported location

## Methodology
- Converted UTC timestamps into hourly and temporal features
- Cleaned and standardized location data
- Filtered to top locations and occupations for interpretability
- Visualized patterns using bar charts, hourly activity plots, and heatmaps

## Key Results
- Posting behavior follows clear daily cycles that vary significantly by region
- Certain occupations exhibit higher late-night activity patterns
- Activity distributions reflect strong time-zone effects across locations

## Tools & Technologies
- Python  
- Pandas  
- Matplotlib / Seaborn / Plotly  

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)
