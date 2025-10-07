###  Overview
So far it involves building an end-to-end insurance fraud detection system that collects and preprocesses claim data, applies machine learning models for fraud prediction, and enhances accuracy through feature engineering. A FastAPI-based web application with interactive dashboards is developed for visualization, while PostgreSQL is integrated for data storage.

###  Dataset Overview

The dataset contains **32 features** in total:
- **6 ordinal features**
- **25 categorical features**
- **1 class label**: `fraud` or `not-fraud`

It includes **15,420 records**, of which only **6% (923 records)** are labeled as fraudulent, indicating a **highly imbalanced dataset**.

###  Environment Setup

Two separate virtual environments are created for better modularity and dependency management:

- **`fraud_ml`** – used for machine learning tasks such as data preprocessing, feature extraction, model training, and evaluation.  
- **`fraud_api`** – used for API development and deployment using FastAPI, integrated with the trained ML model and database.


##  Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Logistic Regression | 0.5597 | 0.9156 | 0.5597 | 0.6658 |
| SVM | 0.8739 | 0.8890 | 0.8739 | 0.8812 |
| Random Forest | 0.9189 | 0.8858 | 0.9189 | 0.9007 |
| Gradient Boosting | 0.9206 | 0.8932 | 0.9206 | 0.9050 |
| MLP Neural Network | 0.9043 | 0.8839 | 0.9043 | 0.8937 |

---

**Best Model:**  Gradient Boosting Classifier  
**F1-Score:** 0.9050  

##  Database Setup 

Created a database named fruad_apidb and two tables are created on the database;

- **claims**
- **risk_assesments**

