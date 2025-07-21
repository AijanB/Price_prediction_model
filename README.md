# 🏘️ Real Estate Price Prediction — Bishkek, Kyrgyzstan 🇰🇬

This project is a machine learning pipeline for predicting real estate prices in Bishkek based on property and location features. It includes data cleaning, feature engineering, clustering, and training of multiple regression models, including CatBoost, Decision Tree, and Linear Regression.

---

## 📊 Dataset Overview

The dataset contains real estate listings with the following types of features:

- **Categorical**: `heating`, `condition`, `doc_quality`, `series`, `building_type`
- **Numerical**: `lat`, `lon`, `rooms`, `total_area`, `floor`, `ceiling_height`, `build_year`, `geo_cluster`, `log_total_area`, `area_per_room`, `is_condition_unknown`

The target variable is the price of the property.

---

## 🧹 Data Cleaning & Feature Engineering

- Removed missing or invalid values
- Log-transformed target variable for modeling
- Created additional features:
  - `log_total_area` = log(1 + `total_area`)
  - `area_per_room` = `total_area` / `rooms`
  - `is_condition_unknown` = flag for unknown conditions

---

## 📌 Correlation Matrix

Shows how numeric features relate to each other and to the target (`usd_price`):

![Correlation Matrix](https://github.com/AijanB/Price_prediction_model/blob/main/images/correlation%20matrix.png?raw=true)

---

## 🧠 Models

### 1. CatBoost Regressor
- Handles categorical features natively
- Tuned hyperparameters with Optuna
- Wrapped with `TransformedTargetRegressor` for log-scaling

### 2. Decision Tree Regressor
- Used `OrdinalEncoder` for categorical features
- Trained on full numerical + encoded categorical dataset

### 3. Linear Model (ElasticNet)
- Pipeline with `OneHotEncoder`, `StandardScaler`, `PolynomialFeatures`
- Wrapped in custom class to handle log target
- Cross-validated with `mean_absolute_percentage_error` (MAPE)

---

## 📈 Model Performance Plots

### 🔵 CatBoost Accuracy Plot  
![CatBoost Accuracy](images/Prediction%20Accuracy%20Plot,%20Catboost.png)

### 🟢 Decision Tree Accuracy Plot  
![DecisionTree Accuracy](images/Prediction%20Accuracy%20Plot%20Desicion%20tree.png)

### 🔴 Linear Model Accuracy Plot  
![Linear Accuracy](images/Prediction%20Accuracy%20Plot%20Linear.png)

---

## 🔍 Feature Importance

### 📌 CatBoost Feature Importance  
![CatBoost FI](images/CatBoost%20Feature%20importance%20.png)

### 🌳 Decision Tree Feature Importance  
![DecisionTree FI](images/DecisionTree%20feature%20importance.png)

### 📐 Linear Model (ElasticNet) Feature Importance  
![Linear FI](images/Feature%20importance%20Linear%20model.png)

---


## 🧪 Evaluation Metrics

- Used R² Score and MAPE
- Visualized Actual vs Predicted Prices with a 45° reference line
- CatBoost shows highest predictive power (R² ≈ 1.00)

---

## 🚀 Conclusion

- **CatBoost** performed the best due to native handling of categorical features and advanced boosting logic.
- **Decision Tree** captured non-linear patterns but showed overfitting tendencies.
- **Linear Model** provided a solid baseline with clear interpretability.

---
