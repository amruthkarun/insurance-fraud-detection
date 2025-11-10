Insurance Claim Fraud Detection and Risk Assessment
pip install -r requirements.txt

REQUIREMENTS Main libraries used:

pandas, numpy, scikit-learn, xgboost seaborn, matplotlib (EDA/visualization) plotly (interactive dashboard) fastapi, uvicorn (web app) sqlalchemy, psycopg2 (PostgreSQL ORM & driver) joblib, pickle (model persistence)

Install : pip install fastapi uvicorn pandas numpy scikit-learn xgboost seaborn matplotlib plotly sqlalchemy psycopg2 joblib google shap google-generativeai python-multipart


DATA PREPROCESSING (prepros.py) Loads insurance_claims.csv Removes null values & irrelevant column (_c39) Encodes categorical columns using LabelEncoder Saves cleaned dataset as processed_test.csv

MODEL TRAINING (model.py) Splits dataset into train/test Tests multiple models (RandomForest, KNN, SVC, XGBoost) Uses XGBoost as final model Evaluates accuracy, precision, recall, F1-score Saves trained model as model.pkl Exports test set for later predictions (test.csv)

WEB APPLICATION (main.py)

FastAPI endpoints:

/ → Homepage with navigation /predict (GET) → Fraud prediction form /predict (POST) → Accepts form input, encodes categorical fields, predicts fraud using model, stores result in PostgreSQL

DATABASE (PostgreSQL)

The app stores user submissions + predictions in a PostgreSQL DB.

Database URL in app.py:

DATABASE_URL = "postgresql://postgres:pw1234@localhost:5432/insurance"

Make sure PostgreSQL is running and database insurance exists

RUNNING THE App

Run preprocessing python prepros.py

Train and save model python model.py

Create a gemini API and the code to main.py

Start FastAPI app uvicorn main:app --reload

Now open:

http://127.0.0.1:8000/ → Homepage http://127.0.0.1:8000/predict → Fraud prediction form http://127.0.0.1:8000/dashboard → Interactive dashboard

DOCKER Containerization

Build the Docker Image

1.Make sure Dockerfile is in the project root, then run: bash: docker build -t insurance-fraud-app .

2.Run the Container bash:docker run -d -p 8000:8000 insurance-fraud-app

App will be available at: http://localhost:8000

DOCKER COMPOSE (App + PostgreSQL)

If you’re using the provided docker-compose.yml, it sets up:

A FastAPI container

A PostgreSQL container

A network bridge for database access

Run both services: bash :docker-compose up --build

This will:

Start PostgreSQL on port 5432

Start FastAPI app on port 8000

Automatically create the insurance database
