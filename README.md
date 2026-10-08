# Customer Churn Prediction System

## 📌 Project Overview

Customer Churn Prediction System is a Machine Learning application that predicts whether a telecom customer is likely to leave the service.

The project follows a complete classification workflow, starting from data collection and cleaning to model training, evaluation, and deployment using FastAPI.

---

## 🎯 Problem Statement

Customer churn can negatively affect a company's revenue and customer retention.

The objective of this project is to build a machine learning classification system that predicts whether a customer will:

- **Churn (Yes)**
- **Not Churn (No)**

This prediction can help businesses identify customers who may be at risk of leaving and take appropriate retention measures.

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn Dataset**.

### Dataset Information

- Records after cleaning: **7,032**
- Features: **20 input features**
- Target variable: **Churn**
- Target classes:
  - Yes
  - No

The original dataset contains customer demographic information, services, contract details, and billing information.

---

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Problem Understanding
2. Data Collection
3. Data Cleaning
4. Data Preprocessing
5. Exploratory Data Analysis (EDA)
6. Feature Engineering
7. Train-Test Split
8. Model Selection
9. Model Training
10. Hyperparameter Optimization
11. Model Evaluation
12. Model Deployment

---

## 🧹 Data Cleaning

The following cleaning operations were performed:

- Removed duplicate records
- Converted `TotalCharges` from text to numeric format
- Handled missing values
- Saved the cleaned dataset

Output file:

```text
data/cleaned_telco_churn.csv