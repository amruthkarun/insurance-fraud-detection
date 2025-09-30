# Developer Documentation

**Project Name** : Insurance Claim Fraud Detection and Risk Assessment
**Develoed By** : Sabarish Rajan
**Tech Stack** : FastAPI, Python(ML/XGBoost/TensorFlow),HTML,JavaScript,PostgreSQL,Joblib, SHAP,GenAI

## 1. Project Overview

This project provides an end-to-end web application for fraud detecttion. It combines:

- **Frontend(HTML/JavaScript)**:User interface for input and visualization.
- **Backend(FastAPI)**:API for model interface, data logging, SHAP explanations.
- **Machine Learning(Python)**:XGBoost Classifier.
- **Explainability**:Generates model interpretability reports.

## 2. Architecture

Frontend (index.html+JS)
|
v
Backend API (FastAPI, my_API.py)
| + --> ML Model (XGBoost, Tensorflow) [model.py saved via joblib]
| + --> Database (PostgreSQL)
| + --> Explainability (SHAP)

## 3. Setup Instructions

### Prerequisites

- Python 3.10+
- PostgreSQL running locally
- Virtual Environment (venv)

### Install Dependencies

Main Libraries:

- fastapi, uvicorn
- xgboost, tensorflow, scikit-learn
- pandas, numpy
- joblib, shap
- psycopg2, sqlalchemy

### Database Setup

1. Create PostgreSQL database Insuauarance_fraud.
2. Define a table claims

### Running the Application

1. Train and load the ML model:
   python Model.py
   (this saves the model as ML_Model.pkl using joblib)
2. Start FastAPI server:
   uvicorn my_API:app --reload
3. Open index.html on browser(make sure API url matches FastAPI endpoint).

## 4. Program Explanation
