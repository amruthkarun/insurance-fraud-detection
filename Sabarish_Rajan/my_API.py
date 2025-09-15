import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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
   'months_as_customer': int64,
    'age': int64, 
    'policy_number': int64, 
    'policy_bind_date': datetime64[ns], 
    'policy_state': object, 
    'policy_csl': object, 
    'policy_deductable': int64, 
    'policy_annual_premium': float64, 
    'umbrella_limit': int64, 
    'insured_zip': int64, 
    'insured_sex': object, 
    'insured_education_level': object, 
    'insured_occupation': object, 
    'insured_hobbies': object, 
    'insured_relationship': object, 
    'capital-gains': int64, 
    'capital-loss': int64, 
    'incident_date': datetime64[ns], 
    'incident_type': object, 
    'collision_type': object, 
    'incident_severity': object, 
    'authorities_contacted': object, 
    'incident_state': object, 
    'incident_city': object, 
    'incident_location': object, 
    'incident_hour_of_the_day': int64,
    'number_of_vehicles_involved': int64, 
    'property_damage': object, 
    'bodily_injuries': int64, 
    'witnesses': int64, 
    'police_report_available': object, 
    'total_claim_amount': int64, 
    'injury_claim': int64, 
    'property_claim': int64, 
    'vehicle_claim': int64, 
    'auto_make': object, 
    'auto_model': object, 
    'auto_year': int64, 


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_fraud(request: Request, data: InputData):
    if not model or not ohe:
        return {"Error": "Model or Encoder not loaded."}
    
    input_df = pd.DataFrame([data.model_dump()])

    input_df = input_df.drop_duplicates()
    input_df = input_df.replace('?', np.nan)

    input_df['authorities_contacted'] = input_df['authorities_contacted'].replace(np.nan, 'No')
    input_df['fraud_reported'] = input_df['fraud_reported'].replace({'Y':1, 'N':0}).astype(int)
    input_df['collision_type'].fillna(input_df['collision_type'].mode()[0])
    input_df['property_damage'].fillna('NO')
    input_df['police_report_available'].fillna('NO')

    input_df = input_df.drop(['age','insured_hobbies','auto_make', 'policy_number','injury_claim','property_claim','vehicle_claim', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip', 'auto_model', 'auto_year'], axis=1)
    cat_cols = input_df.select_dtypes(include=['object']).columns
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = pd.DataFrame(ohe.fit_transform(input_df[cat_cols]), columns = ohe.get_feature_names_out(cat_cols))
    input_df = input_df.drop(cat_cols, axis=1)
    input_df = pd.concat([input_df, encoded_cols], axis=1)

    input_df = pd.concat([input_df, encoded_cols],axis=1)
    input_df = input_df.drop(cat_cols, axis=1)

    Q1 = input_df['policy_annual_premium'].quantile(0.25)
    Q3 = input_df['policy_annual_premium'].quantile(0.75)
    IQR = Q3 - Q1
    filter = (input_df['policy_annual_premium'] >= Q1 - 1.5 * IQR) & (input_df['policy_annual_premium'] <= Q3 + 1.5 * IQR)
    input_df = input_df.loc[filter]

    #Umbrella limit has 50% of the data as 0, hence we create a new binary column
    input_df['umbrella_limit_'] = np.where(input_df['umbrella_limit']==0,0,1)

    input_df = input_df.drop('umbrella_limit', axis=1)

    # total_claim_amount has few outliers, we will remove them
    Q1 = input_df['total_claim_amount'].quantile(0.25)
    Q3 = input_df['total_claim_amount'].quantile(0.75)
    IQR = Q3 - Q1
    filter = (input_df['total_claim_amount']>=Q1 -1.5*IQR) & (input_df['total_claim_amount']<=Q3 + 1.5*IQR)
    input_df = input_df.loc[filter]

    input_df['claim_to_premium_ratio'] = input_df['total_claim_amount']/(input_df['policy_annual_premium']+0.01)
    input_df['severity_of_incident']=input_df['total_claim_amount']/(input_df['witnesses']+0.01)

    fraud_probability = model.predict_proba(input_df)[:,1][0]

    risk_level = "Low Risk"
    if fraud_probability > 0.7:
        risk_level = "High Risk"
    elif fraud_probability > 0.4:
        risk_level = "Medium Risk"

    return {
        "fraud_probability": fraud_probability,
        "risk_level": risk_level
    }

