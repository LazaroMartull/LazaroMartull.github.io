# Clean Water & Sanitation Big Data Analysis (Hadoop & MapReduce)

## Overview
This project applies Big Data processing techniques to analyze global clean water access trends aligned with **UN Sustainable Development Goal 6 (Clean Water & Sanitation)**.

Using Hadoop Streaming and MapReduce, large-scale water access data was processed to compute regional and temporal trends. The results were then visualized and interpreted using Python.

## Problem & Goal
The goal was to analyze large water-access datasets efficiently and answer questions such as:

- How access to basic drinking water changes across regions
- How water access evolves over time by region
- How Big Data tools scale better than traditional single-machine processing

## What I Built
- Custom **Mapper and Reducer scripts** to process large CSV datasets using Hadoop Streaming
- MapReduce logic to compute **average basic water access by region and year**
- A Python notebook to visualize and interpret aggregated Big Data outputs
- A full analytical report connecting technical results to real-world sustainability outcomes

## Big Data Pipeline
- **Input:** Raw global water access CSV data
- **Mapper:** Extracts region, year, and water access values
- **Reducer:** Aggregates values and computes averages per region-year
- **Output:** Clean, summarized datasets for visualization and analysis

## Tools & Technologies
- **Big Data:** Hadoop, MapReduce, Hadoop Streaming
- **Programming:** Python
- **Data Processing:** CSV streaming, key-value aggregation
- **Visualization:** Matplotlib / Pandas
- **Domain:** Sustainable Development Goals (SDG 6)

## Key Takeaways
- Hands-on experience with **distributed data processing**
- Learned how to design MapReduce jobs for aggregation problems
- Understood the tradeoffs between Big Data systems and local analysis
- Connected technical outputs to real-world global development challenges

## Files

📄 **Final Report:** [report.pdf](./report.pdf)

🧠 **MapReduce – Regional Analysis:**  
- [mapper_sdg.py](./SDG_Region/mapper_sdg.py)  
- [reducer_sdg.py](./SDG_Region/reducer_sdg.py)

🧠 **MapReduce – Yearly Analysis:**  
- [mapper_sdg_y.py](./SDG_Yearly/mapper_sdg_y.py)  
- [reducer_sdg_y.py](./SDG_Yearly/reducer_sdg_y.py)

📊 **Analysis & Visuals:**  
- [clean_water_visuals.ipynb](./clean_water_visuals.ipynb)
