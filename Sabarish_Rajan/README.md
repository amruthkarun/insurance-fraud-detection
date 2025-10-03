# Insurance Fraud Detection

This project deals with building a Machine Learning model that detects if an Insurance claim is fraud or not. Why is that important? Simple, insurance fraud accounts for almost $6 billion(Rs. 529 billion) losses for the insurance industry in India, that is about 8.5% of the total industry revenue. This directly leads to an increase in the annual premium payment of an honest insurance policy holders.

This project implements a Machine Learning model to detect fraudulent insurance claims and deploys it as a real-time service. The primary objective is to build an accurate, robust, and interpretable system capable of identifying suspicious activities, allowing insurance companies to minimize payouts on fraudulent claims while ensuring legitimate claims are processed efficiently.

Due to the heavily imbalanced nature of fraud datasets, the final XGBoost model utilizes techniques like SMOTE oversampling and a custom probability threshold optimized for the F1-Score.

## 1. Problem Statement

**Date - 11th September 2025**

Develop a platform using Python to analyze and detect fraudulent insurance claims using open-source datasets and historical claim data. Build a web application with FastAPI, apply machine learning models for fraud detection, and assess risk levels. Implement threading for efficient data processing and integrate PostgreSQL for data management. (e.g., improvisations can be made in feature engineering, incorporating external risk factors, and applying responsible AI concepts for ethical decision-making.)

### Objectives:

Collect, preprocess, and enrich insurance claim data using Python libraries and APIs.
Apply machine learning models (e.g., Logistic Regression, Random Forest) for fraud detection and risk assessment.
Implement feature engineering techniques to enhance model accuracy and reliability.
Develop a web application with FastAPI, featuring interactive dashboards using Plotly or Bokeh.
Integrate PostgreSQL for storing claim data and risk assessments, utilizing threading for efficient data processing tasks.

### GenAI Objective: Use GenAI to extract and process information from unstructured data sources.

## 2. Dataset

**Source**: https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection<br>
**Description**: The dataset used for this analysis contains various features related to insurance claims, including policy details, customer information, accident specifics, and the target variable indicating whether a claim is fraudulent or not.

## 3. Installation and Setup

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

## 4. Project Structure

Frontend (index.html+JS)<br>
|<br>
v<br>
Backend API (FastAPI, my_API.py)<br>
| <br>
+--> ML Model (XGBoost, Tensorflow) [model.py saved via joblib]<br>
| <br>
+--> Database (PostgreSQL)<br>
| <br>
+--> Explainability (SHAP)<br>

## 5. Model Performance and Results

| Model                              | Precision | Recall (Sensitivity) | F1-Score |
| :--------------------------------- | :-------- | :------------------- | :------- |
| **Logistic Regression (Baseline)** | 0.46      | 0.70                 | 0.55     |
| **Descision Tree Classifier**      | 0.46      | 0.70                 | 0.55     |
| **Random Forest Classifier**       | 0.80      | 0.51                 | 0.61     |
| **XGBoost Classifier**             | 0.61      | 0.80                 | 0.70     |
| **Neural Networks**                | 0.64      | 0.71                 | 0.67     |

Key Findings : The XGBoost Classifier was identified as the best performing model. By using SMOTE oversampling and optimizing the final prediction threshold for the F1-Score, the model achieved an F1-Score of 0.70 on the test set. The high Recall (estimated around 0.80) indicates the model is highly effective at minimizing missed fraud cases (False Negatives), which is a critical priority in insurance fraud detection. The final model (ML_Model.pkl) and OneHotEncoder.pkl are saved for deployment.

## 6. Deployment and API service

The model is deployed using a FastAPI service for real-time analysis and explainability.

**Service Details**(my_API.py):

- Framework : FastAPI
- Endpoints: POST /predict
  - Accepts a JSON payload of claim data.
  - Returns the fraud_probability, risk_level, an Returns the fraud_probability, risk_level, an AI-generated Narrative, and a base64 encoded SHAP Waterfall Plot.AI-generated Narrative, and a base64 encoded SHAP Waterfall Plot.
- Data Logging:All new claims processed through the API are automatically inserted into the PostgreSQL table defined by DB_URL.

**Explainability and Narrative**

- SHAP (SHapley Additive exPlanations): Used to calculate feature contributions for each prediction, generating a waterfall plot to visualize which factors drove the result.
- Gemini API: The integrated LLM acts as a "Fraud Analyst AI," consuming the prediction score and the top 5 SHAP drivers to generate a concise, professional justification for the risk prediction, including recommended actions.

**Running the API**

1. Ensure the prerequisites are ready.
2. Start the FastAPI server:
   uvicorn my_API:app --reload
3. Access the web interface at http://127.0.0.1:8000/ to input data and receive predictions.

## 7. Future Works

- GenAI for Unstructured Data Processing: Utilize the GenAI integration to process and extract critical risk factors from unstructured text data, such as police reports, incident narratives, and adjuster notes, generating new, powerful features for the model.
- Responsible AI Concepts: Implement a robust framework for assessing and mitigating model bias. This involves integrating fairness metrics to ensure ethical decision-making and non-discriminatory outcomes across different policyholder demographics.
- External Risk Factor Enrichment: Incorporate external, real-time data sources (e.g., local disaster reports, economic indicators, or social media data) to create more comprehensive risk features.
- Interactive Dashboard Enhancement: Expand the FastAPI application to include interactive dashboards (using Plotly/Bokeh) that display model performance, feature importance drift, and overall fraud trends over time.
- Advanced Scaling and Concurrency: Enhance the data management and processing workflow by fully leveraging Python threading within the FastAPI application for efficient concurrent logging and complex batch risk assessments.
