import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory ="static"), name = 'static')
templates = Jinja2Templates(directory="templates")

try:
    model = joblib.load('ML_Model.pkl')
    ohe = joblib.load('OneHotEncoder.pkl')
    print("Model and Encoder loaded successfully.")
except FileNotFoundError:
    print("Model and/or Encoder not found.")
    model = None
    ohe = None

class InputData(BaseModel):
    months_as_customer: int
    age: int
    policy_number: int
    policy_bind_date: datetime
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
    incident_date: datetime
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


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_fraud(request: Request, data: InputData):
    if not model or not ohe:
        return {"Error": "Model or Encoder not loaded."}
    
    input_df = pd.DataFrame([data.model_dump()])

    input_df = input_df.replace('?', np.nan)
    
    input_df.rename(columns={
    'capital_gains': 'capital-gains',
    'capital_loss': 'capital-loss'
}, inplace=True)


    input_df['authorities_contacted'] = input_df['authorities_contacted'].fillna('No')
    input_df['collision_type'] = input_df['collision_type'].fillna(input_df['collision_type'].mode()[0])
    input_df['property_damage'] = input_df['property_damage'].fillna('NO')
    input_df['police_report_available'] = input_df['police_report_available'].fillna('NO')

    input_df = input_df.drop(['age','insured_hobbies','auto_make', 'policy_number','injury_claim','property_claim','vehicle_claim', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip', 'auto_model', 'auto_year'], axis=1)
    cat_cols = input_df.select_dtypes(include=['object']).columns
    num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns 
    encoded_cols = pd.DataFrame(ohe.transform(input_df[cat_cols]), columns = ohe.get_feature_names_out(cat_cols), index = input_df.index)

    processed_df = input_df.drop(cat_cols, axis = 1)
    processed_df = pd.concat([processed_df, encoded_cols], axis=1)

    
    processed_df['claim_to_premium_ratio'] = processed_df['total_claim_amount']/(processed_df['policy_annual_premium']+0.01)
    processed_df['severity_of_incident']=processed_df['total_claim_amount']/(processed_df['witnesses']+0.01)


    fraud_prediction = model.predict_proba(processed_df)[:,1][0]
    fraud_probability = float(fraud_prediction)
    risk_level = "Low Risk"
    if fraud_probability > 0.7:
        risk_level = "High Risk"
    elif fraud_probability > 0.4:
        risk_level = "Medium Risk"

    return {
        "fraud_probability": fraud_probability,
        "risk_level": risk_level
    }

