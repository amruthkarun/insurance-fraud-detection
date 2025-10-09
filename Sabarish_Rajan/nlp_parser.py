from google import genai
from dotenv import load_dotenv
from google.genai.errors import APIError
import jason
from typing import Dict,Any
load_dotenv()

required_features = [
    "incident_city","incidet_date",'collision_type','incident_severity','authorities_contacted',
    'incident_state', 'incident_city', 'incident_location', 'incident_hour_of_the_day',
    'number_of_vehicles_involved','property_damage','bodily_injuries', 'witnesses',
    'police_report_available', 'total_claim_amount', 'injury_claim','property_claim', 'vehicle_claim',
    'insured_hobbies', 'insured_occupation','insured_education_level', 'insured_relationship','auto_make',
    'auto_model','auto_year'
]

def extraction_prompt(description:str):

    prompt=f"You are an expert insurance claims processor and data extraction agent.
    You are asked to analyse the following incident description provided by the user and 
    extract all the relevant details into a clean JSON object.
    
    **INSTRUCTIONS**
    1. Extract data for the following features {",".join(required_features)}.
    2. Ensure numeric features (like 'total_claim_amount') are clean integers/floats.
    3. If any datapoint is not explicitly mentioned, you MUST set its value to 0(for numbers) or 'null'. Do NOT guess or hallucinate values.
    4. Return only JSON objects. Do not include any conversational texts outside of the JSON block itself.
    
    **USER DESDCRIPTION**
    
    {description}.
    
    ---"

    return prompt

def call_genai_api(prompt:str):
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_KEY:
        try:
            gemini_client=genai.Client(api_key = GEMINI_KEY)
            print("Gemini Loaded Successfully")
        except Exception as e:
            print(f"Error loading Gemini Client:{e}")
    else:
        print("No GenAI key Found.")
        
