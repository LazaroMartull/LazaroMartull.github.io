# Dallas Crime Data: Predicting Crime Locations Using Machine Learning

## Overview
This project builds machine learning models to predict **crime incident locations (premise)** using Dallas Police open data. The task is a **multi-class classification** problem where the target is the reported location type and the inputs are the crime type/category.

## Data
- **Source:** Dallas Open Data Portal (Socrata API)
- **Time Range:** 2014–2024
- **Initial Pull:** 10,000 records via API
- **Final Clean Dataset:** 9,790 rows after cleaning and removing unclear records :contentReference[oaicite:0]{index=0}

**Key fields (renamed):**
- `servyr` → `year`
- `offincident` → `crime`
- `nibrs_crime_category` → `crime_category`
- `premise` → `location` (target) :contentReference[oaicite:1]{index=1}

## Methodology
- Cleaned missing/unclear values (notably in `nibrs_crime_category`) and removed remaining uncertain rows :contentReference[oaicite:2]{index=2}
- One-hot encoded categorical features and trained models on **2014–2023**
- Evaluated performance on a **2024 holdout set**
- Used **3-fold cross-validation** for training evaluation :contentReference[oaicite:3]{index=3}
- Compared results for:
  1) **All location classes**
  2) **Top 5 most frequent locations** (to reduce class imbalance)

## Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest :contentReference[oaicite:4]{index=4}

## Results (Highlights)
### All Locations (2024 Holdout)
- Holdout Accuracy (2024):
  - Logistic Regression: **0.0667**
  - Random Forest: **0.0400**
  - KNN: **0.1867** :contentReference[oaicite:5]{index=5}
- AUC (2024):
  - Logistic Regression: **0.7175**
  - Random Forest: **0.7054**
  - KNN: **0.6107** :contentReference[oaicite:6]{index=6}

### Top 5 Locations (2024 Holdout)
Filtering to the top 5 location classes improved performance substantially:
- Holdout Accuracy (2024):
  - Logistic Regression: **0.4211**
  - Random Forest: **0.4211**
  - KNN: **0.2895** :contentReference[oaicite:7]{index=7}

## Limitations & Improvements
- Performance drops when predicting across many location classes due to **multi-class imbalance and complexity** :contentReference[oaicite:8]{index=8}
- Next steps:
  - collect more data
  - tune hyperparameters
  - try advanced models (SVM, gradient boosting, deep learning) :contentReference[oaicite:9]{index=9}

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)
- 🧪 **Notebook / Code:** [analysis.ipynb](./analysis.ipynb)
- 🎞️ **Slides (optional):** [slides.pdf](./slides.pdf)
