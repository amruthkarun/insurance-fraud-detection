# Developer Documentation

**Project Name** : Insurance Claim Fraud Detection and Risk Assessment<br>
**Developed By** : Sabarish Rajan<br>
**Tech Stack** : FastAPI, Python(ML/XGBoost/TensorFlow),HTML,JavaScript,PostgreSQL,Joblib, SHAP,GenAI<br>

## 1. Project Overview

This project provides an end-to-end web application for fraud detecttion. It combines:

- **Frontend(HTML/JavaScript)**:User interface for input and visualization.
- **Backend(FastAPI)**:API for model interface, data logging, SHAP explanations.
- **Machine Learning(Python)**:XGBoost Classifier.
- **Explainability**:Generates model interpretability reports.

## 2. Architecture

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

**model.py**:

- Prepares the dataset, performs preprocessing(scalling, splitting, OneHOtEncoding)
- Trains ML model (XGBoost)
- Saves model as .pkl for inference.
- Includes performance evaluation: confusion matrix, F1-Score, classification report.

**my_API.py**:

- FastAPI backend
- Exposes endpoints:
  - /get -> takes the values from the user.
  - /predict -> jakes JSON input, runs model, returns prediction, generates SHAP explanation and waterfall plot.
- Logs request and results into PostgreSQL.
- Uses Jinja2 for rendering templates.

**index.html**:

- Frontend interface for users to enter fraud case details.
- Submit forms to FastAPI backend.
- Displays model prediction and explanations.
- Extended with charts.

## 5. How to Run/ Deploy

### Local Deployment

1. Ensure PostgreSQL is running.
2. Start FastAPI with Uvicorn.
3. Open frontend in browser.

## 6. Troubleshooting

- **422 Error(Unprocessable Entity)**: Ensure request body matches Pydantic schema.
- **Model not found**: Verify ML_Model.pkl exists.
- **Database Connection Error**: Update connection string in my_API.py
- **Frontend not Updating**: Refresh/reload the webpage after changes.
