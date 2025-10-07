import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import plotly.express as px
import plotly.io as pio
from fastapi import FastAPI, Request
from api_db import Claim, RiskAssessment, SessionLocal
from model.predict_claim import predict_claim
import threading

app = FastAPI()

pipeline = joblib.load("C:/Users/Veena SP/OneDrive/Desktop/Allianz/Fruad_api/final_pipeline.pkl")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



# @app.post("/predict1")
# async def predict(request: Request):
#     data = await request.form()
#     claim_data = dict(data)

#     thread = threading.Thread(target=process_claim, args=(claim_data,))
#     thread.start()

#     return {"message": "Claim is being processed"} 


@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_fraud(request: Request,Month: str = Form(...),DayOfWeek: str = Form(...),Make: str = Form(...),
    AccidentArea: str = Form(...),Sex: str = Form(...),MaritalStatus: str = Form(...),Age: str = Form(...),
    PolicyType: str = Form(...),VehicleCategory: str = Form(...),VehiclePrice: str = Form(...),
    NumberOfCars: str = Form(...)):
    
    input_data = pd.DataFrame([{"Month": Month,"DayOfWeek": DayOfWeek,"Make": Make,"AccidentArea": AccidentArea,
        "Sex": Sex,"MaritalStatus": MaritalStatus,"Age": Age.replace(",", ""),"PolicyType": PolicyType,
        "VehicleCategory": VehicleCategory,"VehiclePrice": VehiclePrice.replace(",", ""),
        "NumberOfCars": NumberOfCars.replace(",", "")}])

    
    prob = pipeline.predict_proba(input_data)[0][1]
    risk = "High" if prob > 0.5 else "Low"

    fig = px.bar(x=["Fraud Probability"], 
        y=[prob * 100], 
        range_y=[0, 100],
        labels={"y": "Probability (%)"},
        text=[f"{prob*100:.2f}%"]
    )
    fig.update_traces(textposition='outside')
    graph_html = pio.to_html(fig, full_html=False)

    return templates.TemplateResponse("index.html", {"request": request,
        "result": {"prob": round(prob, 2), "risk": risk},"graph": graph_html,
        "Month": Month,"DayOfWeek": DayOfWeek,"Make": Make,"AccidentArea": AccidentArea,
        "Sex": Sex,"MaritalStatus": MaritalStatus,"Age": Age,"PolicyType": PolicyType,"VehicleCategory": VehicleCategory,
        "VehiclePrice": VehiclePrice,"NumberOfCars": NumberOfCars})

def process_claim(claim_data):
    session = SessionLocal()
    try:
        claim = Claim(
            Month=claim_data.get("Month"),
            DayOfWeek=claim_data.get("DayOfWeek"),
            Make=claim_data.get("Make"),
            AccidentArea=claim_data.get("AccidentArea"),
            Sex=claim_data.get("Sex"),
            MaritalStatus=claim_data.get("MaritalStatus"),
            Age=int(claim_data.get("Age", 0)),
            PolicyType=claim_data.get("PolicyType"),
            VehicleCategory=claim_data.get("VehicleCategory"),
            VehiclePrice=claim_data.get("VehiclePrice"),
            NumberOfCars=int(claim_data.get("NumberOfCars", 0))
        )
        session.add(claim)
        session.commit()
        session.refresh(claim)

        prob, risk = predict_claim(claim_data)
        fraud_prob = float(prob)
        risk = str(risk)

        assessment = RiskAssessment(
            claim_id=claim.id,
            fraud_probability=fraud_prob,
            risk_level=risk
        )
        session.add(assessment)
        session.commit()
        return fraud_prob, risk
    finally:
        session.close()

@app.post("/predict1")
async def predict(request: Request):
    data = await request.form()
    claim_data = dict(data)

    # Run directly instead of background thread
    prob, risk = process_claim(claim_data)

    return {
        "fraud_probability": round(prob, 2),
        "risk_level": risk
    }             



   




    
