# 🛒 Big Mart Sales Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting product sales across Big Mart outlets using Machine Learning techniques. The objective is to analyze historical sales data, identify important factors affecting sales, and build predictive models to estimate product sales.

Three machine learning algorithms — **XGBoost, Random Forest, and Support Vector Machine (SVM)** — are implemented and evaluated to compare their predictive performance.

The project covers the complete machine learning pipeline, from data preprocessing and exploratory data analysis to feature engineering, model training, evaluation, and performance comparison.

## 🎯 Objectives

- Analyze Big Mart sales data to identify sales patterns and trends.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and inconsistent data.
- Apply data preprocessing and feature engineering techniques.
- Train multiple machine learning models.
- Evaluate and compare model performance.
- Identify the best-performing model for sales prediction.

## 📂 Project Workflow

1. **Data Collection**
   - Load the Big Mart sales dataset.
   - Understand dataset structure and features.

2. **Exploratory Data Analysis (EDA)**
   - Analyze numerical and categorical variables.
   - Study sales distributions.
   - Identify relationships between features and sales.
   - Visualize important patterns and trends.

3. **Data Preprocessing**
   - Handle missing values.
   - Treat inconsistent categorical values.
   - Encode categorical variables.
   - Perform necessary data transformations.
   - Prepare data for machine learning models.

4. **Feature Engineering**
   - Create and transform relevant features.
   - Extract useful information from existing variables.
   - Select relevant features for model training.
   - Prepare the final feature matrix.

5. **Model Building**

   Three machine learning algorithms were implemented:

   - **XGBoost Regressor:** Gradient boosting algorithm that builds an ensemble of decision trees sequentially.
   - **Random Forest Regressor:** Ensemble learning algorithm that combines multiple decision trees to improve prediction stability.
   - **Support Vector Machine (SVR):** Regression algorithm that uses support vector principles to model relationships between input features and target values.

6. **Model Evaluation & Comparison**
   - Evaluate predictions using regression metrics.
   - Compare the performance of all three models.
   - Identify the most suitable model.

## 🤖 Machine Learning Algorithms

| Algorithm | Description |
|---|---|
| XGBoost | Gradient Boosting-based ensemble learning |
| Random Forest | Bagging-based ensemble learning |
| Support Vector Regression (SVR) | Support Vector-based regression |

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between
