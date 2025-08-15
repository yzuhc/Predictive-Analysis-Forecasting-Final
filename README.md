# Electricity Price Forecasting - Predictive Analytics/Forecasting Final

A comprehensive evaluation of traditional, machine learning, and deep learning approaches for forecasting U.S. electricity prices with rigorous temporal validation.

## Overview
This project analyzes 47 years of U.S. monthly electricity price data (1978-2025) and compares multiple forecasting methodologies including:
- Traditional: ETS, ARIMA, ARIMAX
- Machine Learning: Random Forest, XGBoost
- Deep Learning: Multi-layer Perceptron
- Ensemble methods

## Key Findings
- Enhanced Linear Model achieved best performance (RMSE: 0.0013, MAPE: 0.67%)
- Rigorous temporal validation revealed severe overfitting in Random Forest (6.39x degradation)
- Feature engineering outperformed complex deep learning approaches

## Files
- `final_code.R` - Main analysis script
- `APU000072610.xlsx` - U.S. electricity price data (Bureau of Labor Statistics)

## Requirements
- R 4.0+
- Required packages: readxl, forecast, ggplot2, dplyr, tidyr, gridExtra, lubridate, zoo, randomForest, xgboost, keras

## Usage
1. Install required packages
2. Run `final_code.R`

## Data Source
Bureau of Labor Statistics - Average Price of Electricity to Ultimate Customers by End-Use Sector
