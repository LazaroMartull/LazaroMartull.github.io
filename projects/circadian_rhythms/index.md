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
- Analyzed over **4.5M Twitter records** to identify global circadian activity patterns
- Observed distinct regional posting cycles consistent with time-zone effects
- Identified occupation groups with elevated late-night activity behavior

## Tools & Technologies
- Python  
- Pandas  
- Matplotlib / Seaborn / Plotly  

## Code & Analysis
The full data preprocessing, exploratory analysis, and visualization workflow was implemented in Python using Pandas and visualization libraries. The notebook includes timestamp transformations, location filtering, and multiple plots used in the final report.

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)
- 🧪 **Analysis Notebook:** [Project_Circadian_Rhythms.ipynb](./Project_Circadian_Rhythms.ipynb)
