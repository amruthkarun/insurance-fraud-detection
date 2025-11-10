from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import pickle
import joblib
import sqlalchemy
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import plotly.express as px
import shap
import numpy as np
import google.generativeai as genai
import os


# Load the model and label encoder

model_path = "D:\python\insurance fraud\model.pkl"
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Initialize gemini 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXZEFOMDHC1cuW7V7UnbrkX8xpAMww4_Y"  
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_explanation(model, input_df, prediction_label):
    """
    Generates a human-readable explanation of the model's reasoning
    using SHAP + Gemini 2.5 Flash (free API).
    """
    import shap, numpy as np

    try:
        # Compute SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        shap_contrib = shap_values.values[0]

        # Top 5 contributing features
        top_indices = np.argsort(np.abs(shap_contrib))[::-1][:5]
        top_features = [(input_df.columns[i], shap_contrib[i]) for i in top_indices]
        summary = "\n".join([f"{feat}: {impact:.2f}" for feat, impact in top_features])

        # Prepare a clear prompt for Gemini
        prompt = f"""
        The insurance fraud detection model predicted: '{prediction_label}'.
        The following input features had the strongest influence on the decision:

        {summary}

        Explain this reasoning clearly in simple, transparent lanugage also make it under 200 words,
        make the explaination in two paragraphs, 
        first explaining the shap values and the reason for the outcome,
        second paragraph explains how the model could bring a better outcome what changes could be made.
        """

        # Use Gemini to generate explanation
        model_gen = genai.GenerativeModel("gemini-2.5-flash")
        response = model_gen.generate_content(prompt)

        # Return Gemini’s explanation text
        return response.text.strip()

    except Exception as e:
        return f"Explanation unavailable due to: {e}"


le_path = "D:\python\insurance fraud\label_encoder_ins.pkl"
with open("label_encoder_ins.pkl", "rb") as f:
    label_encoder = joblib.load(f)

# Connect to PostgreSQL - adjust username, password, host, port, dbname
DATABASE_URL = "postgresql://postgres:pw1234@db:5432/insurance"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define table schema
class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True, index=True)
    incident_type = Column(String)
    collision_type = Column(String)
    incident_severity = Column(String)
    policy_annual_premium = Column(Float)
    policy_deductable = Column(Float)
    insured_sex = Column(String)
    insured_education_level = Column(String)
    insured_occupation = Column(String)
    incident_hour_of_the_day = Column(Integer)
    number_of_vehicles_involved = Column(Integer)
    property_damage = Column(String)
    bodily_injuries = Column(Integer)
    total_claim_amount = Column(Float)
    auto_model = Column(String)
    auto_year = Column(Integer)
    months_as_customer = Column(Integer)
    capital_gains = Column(Float)
    capital_loss = Column(Float)
    insured_hobbies = Column(String)
    authorities_contacted = Column(String)
    witnesses = Column(Integer)
    police_report_available = Column(String)
    injury_claim = Column(Float)
    property_claim = Column(Float)
    prediction = Column(String)

Base.metadata.create_all(bind=engine)

app = FastAPI()

#inputstructure 
class ClaimData(BaseModel):
    incident_type: str
    collision_type: str
    incident_severity: str
    policy_annual_premium: float
    policy_deductable: float
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    property_damage: str
    bodily_injuries: int
    total_claim_amount: float
    auto_model: str
    auto_year: int
    months_as_customer: int
    capital_gains: float
    capital_loss: float
    insured_hobbies: str
    authorities_contacted : str
    witnesses: int
    police_report_available: str
    injury_claim: float
    property_claim: float
    

# prediction
@app.get("/predict", response_class=HTMLResponse)
def get_predict_form():
    
    form_html = """
    
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f9f9f9;
        }
        h2 {
            text-align: center;
            font-size: 32px;
            margin-bottom: 30px;
        }
        .form-container {
            display: flex;
            justify-content: center;
        }
        .form-box {
            display: flex;
            flex-wrap: wrap;
            max-width: 900px;
            background: #fff;
            padding: 20px 40px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .form-group {
            width: 45%;
            margin: 10px 2.5%;
        }
        label {
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        .submit-btn {
            width: 100%;
            text-align: center;
            margin-top: 20px;
        }
        .submit-btn input {
            background: #007BFF;
            color: #fff;
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        .submit-btn input:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <h2>Insurance Fraud Prediction</h2>
    <div class="form-container">
        <form method="post" action="/predict" class="form-box">
            
            <div class="form-group">
                <label>Incident Type:</label>
                <input type="text" name="incident_type">
            </div>
            
            <div class="form-group">
                <label>Collision Type:</label>
                <input type="text" name="collision_type">
            </div>
            
            <div class="form-group">
                <label>Incident Severity:</label>
                <input type="text" name="incident_severity">
            </div>
            
            <div class="form-group">
                <label>Policy Annual Premium:</label>
                <input type="number" step="0.01" name="policy_annual_premium">
            </div>
            
            <div class="form-group">
                <label>Policy Deductable:</label>
                <input type="number" step="0.01" name="policy_deductable">
            </div>
            
            <div class="form-group">
                <label>Insured Sex:</label>
                <select name="insured_sex">
                    <option value="MALE">MALE</option>
                    <option value="FEMALE">FEMALE</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Insured Education Level:</label>
                <input type="text" name="insured_education_level">
            </div>
            
            <div class="form-group">
                <label>Insured Occupation:</label>
                <input type="text" name="insured_occupation">
            </div>
            
            <div class="form-group">
                <label>Incident Hour Of The Day:</label>
                <input type="number" name="incident_hour_of_the_day">
            </div>
            
            <div class="form-group">
                <label>Number Of Vehicles Involved:</label>
                <input type="number" name="number_of_vehicles_involved">
            </div>
            
            <div class="form-group">
                <label>Property Damage:</label>
                <select name="property_damage">
                    <option value="YES">YES</option>
                    <option value="NO">NO</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Bodily Injuries:</label>
                <input type="number" name="bodily_injuries">
            </div>
            
            <div class="form-group">
                <label>Total Claim Amount:</label>
                <input type="number" step="0.01" name="total_claim_amount">
            </div>
            
            <div class="form-group">
                <label>Auto Model:</label>
                <input type="text" name="auto_model">
            </div>
            
            <div class="form-group">
                <label>Auto Year:</label>
                <input type="number" name="auto_year">
            </div>
            
            <div class="form-group">
                <label>Months as Customer:</label>
                <input type="number" name="months_as_customer">
            </div>
            
            <div class="form-group">
                <label>Capital Gains:</label>
                <input type="number" step="0.01" name="capital_gains">
            </div>
            
            <div class="form-group">
                <label>Capital Loss:</label>
                <input type="number" step="0.01" name="capital_loss">
            </div>
            
            <div class="form-group">
                <label>Insured Hobbies:</label>
                <input type="text" name="insured_hobbies">
            </div>
            
            <div class="form-group">
                <label>Authorities Contacted:</label>
                <input type="text" name="authorities_contacted">
            </div>
            
            <div class="form-group">
                <label>Witnesses:</label>
                <input type="number" name="witnesses">
            </div>
            
            <div class="form-group">
                <label>Police Report Available:</label>
                <select name="police_report_available">
                    <option value="YES">YES</option>
                    <option value="NO">NO</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Injury Claim:</label>
                <input type="number" step="0.01" name="injury_claim">
            </div>
            
            <div class="form-group">
                <label>Property Claim:</label>
                <input type="number" step="0.01" name="property_claim">
            </div>
            
            <div class="submit-btn">
                <input type="submit" value="Predict Fraud">
            </div>
            
        </form>
    </div>
</body>
</html>
"""



    return form_html

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f7fa;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            background: #fff;
            padding: 40px 60px;
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 40px;
            margin-bottom: 20px;
            color: #333;
        }
        p {
            margin: 15px 0;
            font-size: 18px;
        }
        a {
            color: #007BFF;
            text-decoration: none;
            font-weight: bold;
            transition: 0.3s;
        }
        a:hover {
            color: #0056b3;
            text-decoration: underline;
        }
        .links {
            margin-top: 20px;
        }
        .btn {
            display: inline-block;
            margin: 10px;
            padding: 12px 25px;
            background: #007BFF;
            color: #fff;
            border-radius: 8px;
            text-decoration: none;
            font-size: 16px;
            transition: 0.3s;
        }
        .btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Insurance Claims</h1>
        <p>Welcome! Choose an option below:</p>
        <div class="links">
            <a href='/dashboard' class="btn">View Dashboard</a>
            <a href='/predict' class="btn">Fraud Predictor</a>
        </div>
    </div>
</body>
</html>
"""

# Accept form post
@app.post("/predict", response_class=HTMLResponse)
async def post_predict(
    incident_type: str = Form(...),
    collision_type: str = Form(...),
    incident_severity: str = Form(...),
    policy_annual_premium: float = Form(...),
    policy_deductable: float = Form(...),
    insured_sex: str = Form(...),
    insured_education_level: str = Form(...),
    insured_occupation: str = Form(...),
    incident_hour_of_the_day: int = Form(...),
    number_of_vehicles_involved: int = Form(...),
    property_damage: str = Form(...),
    bodily_injuries: int = Form(...),
    total_claim_amount: float = Form(...),
    auto_model: str = Form(...),
    auto_year: int = Form(...),
    months_as_customer: int = Form(...),
    capital_gains: float = Form(...),
    capital_loss: float = Form(...),
    insured_hobbies: str = Form(...),
    authorities_contacted: str = Form(...),
    witnesses: int = Form(...),
    police_report_available: str = Form(...),
    injury_claim: float = Form(...),
    property_claim: float = Form(...)
):
    # Encode categorical inputs using label encoder 
    try:
        incident_type = label_encoder.transform([incident_type])[0]
    except Exception:
        incident_type = -1  #  handle unknown category appropriately
    

    try:
        collision_type = label_encoder.transform([collision_type])[0]
    except Exception:
        collision_type = -1 

    try:
        incident_severity = label_encoder.transform([incident_severity])[0]
    except Exception:
        incident_severity = -1 

    try:
        insured_sex = label_encoder.transform([insured_sex])[0]
    except Exception:
        insured_sex = -1 

    try:
        insured_education_level = label_encoder.transform([insured_education_level])[0]
    except Exception:
        insured_education_level = -1 

    try:
        insured_occupation = label_encoder.transform([insured_occupation])[0]
    except Exception:
        insured_occupation = -1 

    try:
        property_damage = label_encoder.transform([property_damage])[0]
    except Exception:
        property_damage = -1 

    try:
        auto_model = label_encoder.transform([auto_model])[0]
    except Exception:
        auto_model = -1 

    try:
        insured_hobbies = label_encoder.transform([insured_hobbies])[0]
    except Exception:
        insured_hobbies = -1

    try:
        authorities_contacted = label_encoder.transform([authorities_contacted])[0]
    except Exception:
        authorities_contacted = -1

    try:
        police_report_available = label_encoder.transform([police_report_available])[0]
    except Exception:
        police_report_available = -1
    

    feature_names = ['months_as_customer', 'policy_deductable', 'policy_annual_premium',
 'insured_sex','insured_education_level','insured_occupation',
 'insured_hobbies','capital_gains','capital_loss','incident_type',
 'collision_type','incident_severity','authorities_contacted',
 'incident_hour_of_the_day','number_of_vehicles_involved',
 'property_damage','bodily_injuries','witnesses',
 'police_report_available','total_claim_amount','injury_claim',
 'property_claim','auto_model','auto_year']
    
    #  input vector for model prediction
    input_values = [
    months_as_customer, policy_deductable, policy_annual_premium,
    insured_sex, insured_education_level, insured_occupation,
    insured_hobbies, capital_gains, capital_loss, incident_type,
    collision_type, incident_severity, authorities_contacted,
    incident_hour_of_the_day, number_of_vehicles_involved,
    property_damage, bodily_injuries, witnesses,
    police_report_available, total_claim_amount, injury_claim,
    property_claim, auto_model, auto_year
]


    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Run prediction
    pred = model.predict(input_df)[0]
    pred_str = "Possible Fraud" if pred == 1 else "Legitimate"

        # Get fraud probability
    if hasattr(model, "predict_proba"):
      fraud_prob = model.predict_proba(input_df)[0][1] * 100  # percentage
    else:
      fraud_prob = None



    # Generate explainability output
    explanation = generate_explanation(model, input_df, pred_str)

    # Save user data + prediction to DB
    db = SessionLocal()
    db_claim = Claim(
        incident_type=incident_type,
        collision_type=collision_type,
        incident_severity=incident_severity,
        policy_annual_premium=policy_annual_premium,
        policy_deductable=policy_deductable,
        insured_sex=insured_sex,
        insured_education_level=insured_education_level,
        insured_occupation=insured_occupation,
        incident_hour_of_the_day=incident_hour_of_the_day,
        number_of_vehicles_involved=number_of_vehicles_involved,
        property_damage=property_damage,
        bodily_injuries=bodily_injuries,
        total_claim_amount=total_claim_amount,
        auto_model=auto_model,
        auto_year=auto_year,
        months_as_customer=months_as_customer,
        capital_gains=capital_gains,
        capital_loss=capital_loss,
        insured_hobbies=insured_hobbies,
        authorities_contacted=authorities_contacted,
        witnesses=witnesses,
        police_report_available=police_report_available,
        injury_claim=injury_claim,
        property_claim=property_claim,
        prediction=pred_str
    )
    db.add(db_claim)
    db.commit()
    db.close()

    # Return result page
    result_html = f"""
    <html>
    <body>
        <h2>Prediction Result</h2>
        <p><b>Prediction:</b> {pred_str}</p>
        <p><b>Fraud Probability:</b> {fraud_prob:.2f}%</p>
        <h3>Model Explainability (GEMINI AI)</h3>
        <div style='background:#f0f0f0;padding:15px;border-radius:10px;width:60%;'>
         {explanation}
        </div>
        <p><a href="/predict">Back</a></p>
        <p><a href="/">Home</a></p>
    </body>
    </html>
    """
    return result_html
     
df = pd.read_csv("insurance_claims.csv")

#dashboard

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    # Chart 1: Total claim amount by incident type - Bar chart
    chart1 = px.bar(
        df.groupby("incident_type")["total_claim_amount"].sum().reset_index(),
        x="incident_type",
        y="total_claim_amount",
        title="Total Claim Amount by Incident Type"
    )
    
    # Chart 2: Number of claims by incident severity - Pie chart
    severity_counts = df["incident_severity"].value_counts().reset_index()
    severity_counts.columns = ["incident_severity", "count"]
    chart2 = px.pie(
        severity_counts,
        values="count",
        names="incident_severity",
        title="Claims Distribution by Incident Severity"
    )
    
    # Chart 3: Average claim amount by policy state
    avg_claim_by_state = df.groupby("policy_state")["total_claim_amount"].mean().reset_index()
    chart3 = px.choropleth(
        avg_claim_by_state,
        locations="policy_state",
        locationmode="USA-states",
        color="total_claim_amount",
        color_continuous_scale="Viridis",
        scope="usa",
        title="Average Claim Amount by Policy State"
    )

    
    
    # Combine the charts into one HTML string for the dashboard page
    dashboard_html = f"""
    <html>
    <head>
        <title>Insurance Claims Dashboard</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
    </head>
    <body>
        <h1>Insurance Claims Dashboard</h1>
        <div>{chart1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div>{chart2.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div>{chart3.to_html(full_html=False, include_plotlyjs=False)}</div>
       
        <p>Go to <a href='/'>home</a>.</p>


    </body>
    </html>
    """

    return HTMLResponse(content=dashboard_html)

