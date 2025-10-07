import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import date
import os
import io
import matplotlib.pyplot as plt
import base64
import psycopg2
from sqlalchemy import create_engine
import json
import shap
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from data_prep import DataPreprocessor
load_dotenv()

app = FastAPI()

processor = DataPreprocessor()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client initialized successfully from .env file.")
    except Exception as e:
        print(f"Failed to initialize Gemini Client. Check API Key: {e}")
        gemini_client = None
else:
    print("GEMINI_API_KEY not found in environment or .env file.")
    gemini_client = None

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)
table_name = 'claims'

try:
    model = joblib.load("ML_Model.pkl")
    ohe = joblib.load("OneHotEncoder.pkl")
    print("Model and Encoder loaded successfully.")
except FileNotFoundError:
    print("Model and/or Encoder not found.")
    model = None
    ohe = None

try:
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    print("SHAP Explainer Initialized")
except Exception as e:
    print("Error initializing explainer:{e}")
    explainer = None


class InputData(BaseModel):
    months_as_customer: int
    age: int
    policy_number: int
    policy_bind_date: date
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
    incident_date: date
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_location: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    property_damage: str
    bodily_injuries: int
    witnesses: int
    police_report_available: str
    total_claim_amount: int
    injury_claim: int
    property_claim: int
    vehicle_claim: int
    auto_make: str
    auto_model: str
    auto_year: int

def insert_new_claims(new_claim_data:pd.DataFrame, table_name:str, db_engine):
    print('Attempting to insert new record into the database.')
    try:
        new_claim_data.to_sql(
            name=table_name,
            con=db_engine,
            if_exists='append',
            index=False
        )
        print('Succesfully added.')
    except Exception as e:
        print(f'Encountered an error: {e}')

def generate_narrative(
    risk_level: str, probability: float, top_drivers: pd.Series, raw_data: dict
):

    if not gemini_client:
        return "GenAI services not available"

    prompt_lines = [
        "You are Fraud Analyst AI. Generate a concise, professional justification for the prediction of this insurance claim.",
        f"The predictor model returned a **{risk_level}** status with a probability of **{probability:.1%}** of fraud.",
        "Analyze the top 5 most influential factors and their actual values to explain why the model flagged the claim.",
        "The influential factors are:"
    ]

    for feature, shap_value in top_drivers.items():

        original_feature = feature.split("_")[0] if "_" in feature else feature

        feature_value = raw_data.get(feature, raw_data.get(original_feature, "N/A"))
        if feature_value == "N/A" and "_" in feature:
            feature_value = 1

        influence = (
            "pushed the score **towards fraud**"
            if shap_value > 0
            else "pulled the score **away from fraud**"
        )
        prompt_lines.append(
            f"- **Feature:** `{feature}` (Actual Value: `{feature_value}`). Its influence {influence} (SHAP Value: `{shap_value:.4f}`)."
        )

    final_prompt = (
        " ".join(prompt_lines)
        + " Focus on professional analysis of these factors and conclude with recommendations for actions. In less than 20 lines. List only the top 3 influential factors without listing its shap values. Write the recomendations as a seperate paraghraph. Highlight the recomendation part. Do not print the important features as it is, instead write them in a user friendly manner. Do not use ** at all. Instead bolden the words in **."
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=final_prompt
        )
        return response.text
    except APIError as e:
        return "API Error: Could not generate narrative."
    except Exception as e:
        print(f"Error occured:{e}")
        return "Unexpected error encountered. Cannot generate narrative."


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_fraud(request: Request, data: InputData):
    if not model or not ohe:
        return {"Error": "Model or Encoder not loaded."}

    input_df = pd.DataFrame([data.model_dump()])
    
    

    input_df.rename(
        columns={"capital_gains": "capital-gains", "capital_loss": "capital-loss"},
        inplace=True,
    )
    
    new_claim_data = input_df

    input_df = processor._clean(input_df)

    cat_cols = input_df.select_dtypes(include=["object"]).columns
    num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
    encoded_cols = pd.DataFrame(
        ohe.transform(input_df[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=input_df.index,
    )

    processed_df = input_df.drop(cat_cols, axis=1)
    processed_df = pd.concat([processed_df, encoded_cols], axis=1)

    processed_df["claim_to_premium_ratio"] = processed_df["total_claim_amount"] / (
        processed_df["policy_annual_premium"] + 0.01
    )
    processed_df["severity_of_incident"] = processed_df["total_claim_amount"] / (
        processed_df["witnesses"] + 0.01
    )

    # print(processed_df.columns)
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
    new_claim_data['fraud_reported'] = fraud_reported
    if explainer and risk_level in ["Low Risk", "Medium Risk", "High Risk"]:
        scaled_df_for_shap = model.named_steps["scaler"].transform(processed_df)
        shap_value = explainer.shap_values(scaled_df_for_shap)[0]
        feature_names = processed_df.columns

        shap_series = pd.Series(shap_value, index=feature_names)

        top_drivers_abs = shap_series.abs().sort_values(ascending=False).head(5)
        top_drivers_with_sign = shap_series.loc[top_drivers_abs.index]

        genai_narrative = generate_narrative(
            risk_level, fraud_probability, top_drivers_with_sign, data.model_dump()
        )
    insert_new_claims(new_claim_data, table_name, engine)
    

    waterfall_plot_base64 = None
    try:
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_value, feature_names=processed_df.columns, show=False
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        waterfall_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()
    except Exception as e:
        print(f"Error generating SHAP waterfall: {e}")
    
    return {
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "Narrative": genai_narrative,
        "waterfall_plot": waterfall_plot_base64
    }
    