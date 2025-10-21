import joblib #For loading the trained model and encoder to a pkl file
from fastapi import FastAPI, Request, BackgroundTasks #Creating the API, getting the values from the user from frontend, run tasks in the background so that the prediction logic can run smoothly
from pydantic import BaseModel # Defining the schema of the input
import numpy as np
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import date
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import psycopg2 #PostgreSQL database connection
from sqlalchemy import create_engine #Inseeting data into the database
import json
import shap #Explainer library
from google import genai #GenAI client Library
from google.genai.errors import APIError
from dotenv import load_dotenv
from data_prep import DataPreprocessor #Data Preprocessing class
from nlp_parser import extract_incident_data #NLP Parser class
from insert_to_database import DatabaseInsertion #Database Insertion class
from narrative_generation import GenerateNarrative
import threading
from anyio import to_thread #For running blocking database operations in a separate thread
import asyncio

load_dotenv()

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Gemini API Key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client initialized successfully from .env file.")
    except Exception as e:
        print(f"Failed to initialize Gemini Client. Check API Key:{e}")
        gemini_client = None
else:
    print("GEMINI_API_KEY not found in environment or .env file.")
    gemini_client = None
# Load the model and preprocessor
try:
    model = joblib.load("ML_Model.pkl")
    preprocessor_ = joblib.load("data_preprocessor.pkl")
    feat_order = joblib.load("Feature_order.pkl")
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading Model or Encoder{e}")
    model = None
    ohe = None
    feat_order= None

#Object creation
insertion = DatabaseInsertion() 
processor = DataPreprocessor()
extraction_agent = extract_incident_data()
shap_explainer = GenerateNarrative(model_ = model, gemini_client_ = gemini_client)

# Define input data schema
class InputData(BaseModel):
    months_as_customer: int
    age: int 
    policy_number: int 
    policy_bind_date: date = None
    policy_state: str 
    policy_csl: str
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: int
    insured_zip: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: int
    capital_loss: int

    description: str = None

# Define API endpoints

# Root endpoint to serve the HTML page
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request, data: InputData, background_tasks:BackgroundTasks):
    # Check if model and preprocessor are loaded
    if not model or not preprocessor_:
        return {"Error:Model or Encoder not loaded."}
    # Getting the input as a dictionary
    feat_dict = data.model_dump()
    # Get the features extracted from descriptionusing NLP Parser
    extracted_feat_dict = extraction_agent.extract_data_from_des(data.description)

    if not extracted_feat_dict:
        return {"Error:Could not extract features from description."}
    
    combined_features = {**feat_dict, **extracted_feat_dict}

    if 'description' in combined_features:
        del combined_features['description']

    input_df = pd.DataFrame([combined_features])

    input_df.rename(
        columns={"capital_gains": "capital-gains", "capital_loss": "capital-loss"},
        inplace=True
    )
    
    new_claim_data = input_df
    #Process the input data
    input_df = processor._clean(input_df)
    processed_df = preprocessor_.preprocess_incident_data(input_df)
    processed_df = input_df.reindex(columns = feat_order, fill_value = 0.0)
    
    #Feature Engineering
    processed_df["claim_to_premium_ratio"] = processed_df["total_claim_amount"] / (
        processed_df["policy_annual_premium"] + 0.01
    )
    processed_df["severity_of_incident"] = processed_df["total_claim_amount"] / (
        processed_df["witnesses"] + 0.01
    )

    processed_df = processed_df.drop('fraud_reported', axis = 1)

    #Predicting Fraud Probability
    fraud_prediction = model.predict_proba(processed_df)[:, 1][0]
    fraud_probability = float(fraud_prediction)
    risk_level = "Low Risk"
    if fraud_probability > 0.7:
        risk_level = "High Risk"
    elif fraud_probability > 0.4:
        risk_level = "Medium Risk"
    fraud_reported = False
    if fraud_probability > 0.5:
        fraud_reported = True
    genai_narrative = "Narrative Skipped or not generated"

    # Updating the new claim data with fraud reported
    new_claim_data['fraud_reported'] = fraud_reported

    #Generate SHAP explanations and Narrative
    
    args_for_narrative = (
        risk_level, 
        fraud_probability, 
        data.model_dump(), 
        processed_df
    )

    try:
        genai_narrative = await to_thread.run_sync(
            shap_explainer.generate_narrative,
            *args_for_narrative
        )
        print('Narrative generated successfully')
    except Exception as e:
        print(f'Error during ai generation:{e}')
        genai_narrative = "Error generating narrative."

    insert = False

    try:
        background_tasks.add_task(insertion.insert_new_claims, new_claim_data)
        print("Insertion added to background tasks.")
        insert = True
    except Exception as e:
        print(f"Error occuerd while insertion:{e}")

    #Generate SHAP Waterfall plot
    waterfall_plot_base64 = None
    try:
        expected_value = shap_explainer.expected_value[0]
        shap_value = shap_explainer.shap_values
        feature_names = shap_explainer.feature_names
        if shap_value is not None and feature_names is not None:

            shap.plots._waterfall.waterfall_legacy(
                expected_value, 
                shap_value, 
                feature_names=feature_names, 
                show=False
            )

            def save_fig(buf):
                plt.savefig(buf, format='png', bbox_inches = 'tight')
                plt.close()
            plt_buf = io.BytesIO()
            try:
                await to_thread.run_sync(save_fig, plt_buf)

                plt_buf.seek(0)
                waterfall_plot_base64 = base64.b64encode(plt_buf.read()).decode("utf-8")
                plt_buf.close()
                print('Plot saved successfully.')
            except Exception as e:
                print(f"Error saving SHAP waterfall plot:{e}")

    except Exception as e:
        print(f"Error generating SHAP waterfall: {e}")
    
    # Return the prediction and explanations
    return {
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "Narrative": genai_narrative,
        "waterfall_plot": waterfall_plot_base64,
        "Insertion_Done":insert
    }