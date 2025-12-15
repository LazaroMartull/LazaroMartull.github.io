# Dallas Crime Data: Predicting Crime Locations Using Machine Learning

## Overview
This project builds machine learning models to predict **crime incident locations (premise)** using Dallas Police open data. The task is a **multi-class classification** problem where the target variable is the reported location type and the inputs are the crime type and category.

## Data
- **Source:** Dallas Open Data Portal (Socrata API)
- **Time Range:** 2014–2024
- **Initial Pull:** 10,000 records via API
- **Final Clean Dataset:** 9,790 records after cleaning and removing unclear values

**Key fields (renamed):**
- `servyr` → `year`
- `offincident` → `crime`
- `nibrs_crime_category` → `crime_category`
- `premise` → `location` (target variable)

## Methodology
- Removed missing or unclear values (notably in `nibrs_crime_category`)
- One-hot encoded categorical features
- Trained models on **2014–2023** data
- Evaluated performance on a **2024 holdout set**
- Used **3-fold cross-validation** during training
- Compared results for:
  - All location classes
  - Top 5 most frequent locations (to reduce class imbalance)

## Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

## Results (Highlights)

### All Locations (2024 Holdout)
**Holdout Accuracy:**
- Logistic Regression: **0.0667**
- Random Forest: **0.0400**
- KNN: **0.1867**

**AUC:**
- Logistic Regression: **0.7175**
- Random Forest: **0.7054**
- KNN: **0.6107**

### Top 5 Locations (2024 Holdout)
Filtering to the top five most common location classes significantly improved performance:

**Holdout Accuracy:**
- Logistic Regression: **0.4211**
- Random Forest: **0.4211**
- KNN: **0.2895**

## Limitations & Future Work
- Performance decreases when predicting across many location classes due to **multi-class imbalance and problem complexity**
- Potential improvements include:
  - Collecting additional data
  - Hyperparameter tuning
  - Exploring advanced models such as SVMs, gradient boosting, or deep learning

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)
- 🧪 **Notebook / Code:** [analysis.ipynb](./analysis.ipynb)
