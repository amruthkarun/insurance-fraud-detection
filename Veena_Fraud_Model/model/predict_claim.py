
import pandas as pd
import joblib

pipeline = joblib.load("C:/Users/Veena SP/OneDrive/Desktop/Allianz/Fruad_api/randomforest_model.pkl")

def predict_claim(form_data: dict):
    input_df = pd.DataFrame([form_data])
    prob = pipeline.predict_proba(input_df)[0][1]
    risk = "High" if prob > 0.5 else "Low"
    return round(prob, 2), risk
