# Task3_XyloFi
# 📈 End-to-End Sales Forecasting & Demand Intelligence System

## Overview

This project presents a complete Sales Forecasting and Demand Intelligence solution developed using Python and Machine Learning.

The system analyzes historical retail sales, identifies seasonal trends, forecasts future demand, detects abnormal sales behavior, segments products based on demand patterns, and presents the insights through an interactive Streamlit dashboard.

---

## Objectives

- Analyze historical sales performance
- Forecast future sales using multiple forecasting techniques
- Detect unusual sales spikes and drops
- Segment products based on demand characteristics
- Build an interactive business dashboard

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

## Machine Learning Models

### Forecasting

- SARIMA
- Facebook Prophet
- XGBoost Regressor

### Anomaly Detection

- Isolation Forest
- Z-Score Analysis

### Product Segmentation

- K-Means Clustering
- PCA Visualization

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Time Series Decomposition
6. Stationarity Testing (ADF)
7. Forecasting Models
8. Model Comparison
9. Category & Region Analysis
10. Anomaly Detection
11. Product Segmentation
12. Streamlit Dashboard

---

## Project Structure

```
SalesForecasting_ParitGupta/

│── analysis.ipynb
│── app.py
│── train.csv
│── requirements.txt
│── summary.pdf
│── README.md
│── charts/
```

---

## Results

- Technology category generated the highest sales.
- West region recorded the strongest overall sales performance.
- Monthly sales showed an increasing trend with seasonal variations.
- XGBoost achieved the best forecasting performance among the evaluated models.
- Isolation Forest successfully detected abnormal sales spikes and drops.
- Product demand segmentation grouped sub-categories into four business clusters.

---

## Business Recommendations

- Deploy XGBoost for production forecasting.
- Increase inventory for high-demand products.
- Monitor anomalies to identify promotional or operational events.
- Use demand clusters for inventory optimization.
- Continuously monitor seasonal sales patterns for better planning.

---

## Challenges Faced

- Time-series preprocessing
- Feature engineering for forecasting
- SARIMA parameter tuning
- Forecast model comparison
- Stationarity testing
- Anomaly detection tuning
- Product clustering interpretation
- Streamlit dashboard integration

---

## Future Improvements

- Deep Learning models (LSTM)
- Real-time forecasting
- Automated model retraining
- Cloud deployment
- Power BI integration
- Interactive KPI dashboard

---

## Author

**Parit Gupta**

Third Year Undergraduate Student

Machine Learning | Data Science | Python
