from google import genai
from dotenv import load_dotenv
#from google.genai.errors import APIError
import json
from typing import Dict,Any
import requests
import time
import os

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    print("Gemini loaded successfully.")
else:
    print('Gemini Key not found.')

required_features = [
    'incident_city','incident_date','collision_type','incident_severity','authorities_contacted',
    'incident_state', 'incident_location', 'incident_hour_of_the_day',
    'number_of_vehicles_involved','property_damage','bodily_injuries', 'witnesses',
    'police_report_available', 'total_claim_amount', 'injury_claim','property_claim', 'vehicle_claim',
    'insured_hobbies', 'insured_occupation','insured_education_level', 'insured_relationship','auto_make',
    'auto_model','auto_year'
]

numeric_features = [
    'incident_hour_of_the_day','number_of_vehicles_involved','bodily_injuries','witnesses','total_claim_amount',
    'injury_claim','property_claim','vehicle_claim','auto_year'
]

def extraction_prompt(description:str):

    prompt=f"""
    You are an expert insurance claims processor and data extraction agent.
    You are asked to analyse the following incident description provided by the user and 
    extract all the relevant details into a clean JSON object.
    
    **INSTRUCTIONS**
    1. Extract data for the following features {",".join(required_features)}.
    2. Ensure numeric features (like 'total_claim_amount') are clean integers/floats.
    3. If any datapoint is not explicitly mentioned, you MUST set its value to 0(for numbers) or 'null'. Do NOT guess or hallucinate values.
    4. Return only JSON objects. Do not include any conversational texts outside of the JSON block itself.
    
    **USER DESCRIPTION**
    
    {description}.
    
    ---
    """

    return prompt

def schema_gen(all_feat:list, num_feat:list):

    properties={}
    for feat in all_feat:
        prop_type = 'integer' if feat in num_feat else 'string'

        properties[feat]={"type":prop_type}

    return{
        "type":"object",
        'properties':properties,
        'property_ordering':all_feat
    }


def call_genai_api(prompt:str,api_key:str):
    model_name = 'gemini-2.5-flash'
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    max_tries = 5
    response_schema = schema_gen(required_features, numeric_features)

    payload={
        'contents':[{'parts':[{'text':prompt}]}],
        'tools':[{'google_search':{}}],
        'generationConfig': {
            'responseMimeType':'application/json',
            'responseSchema':response_schema
        },
        'systemInstruction': {
            'parts':[{
            'text' : (
                "You are an expert insurance claims processor and data extractor."
                "Your sole task is to analyze the user's claim data description and extract relevant information"
                "into a structured JSON object defined in the schema."
                "If any data point is not specified then you MUST set its value to 'null' or 0(for numerical values)."
                "DO NOT HALLUCINATE VALUES."
            )
            }]
        },
    }

    json_string =""

    for tries in range(max_tries):
        try:
            response = requests.post(api_url, headers = {'Content-Type':'application/json'}, json = payload)
            response.raise_for_status()
            result = response.json()

            candidate = result.get('candidates',[{}])[0]
            if candidate:
                json_string = candidate.get('content',{}).get('parts',[])[0].get('text')
                if json_string:
                    return json_string

        except Exception as e:
            print(f"API call failed on {tries+1}:{e}")

        if tries < (max_tries-1):
            delay = 2**tries
            time.sleep(delay)
            print(f"Reloading API in {delay} seconds.")
        
    print(f"API call failed after {max_tries} attempts.")

    return "{}"

def parse_and_validate_json(json_string:str):

    data = {}

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM:{e}")
        return {}
    
    final_data = {}

    for feat in required_features:
        value = data.get(feat)
        
        if feat in numeric_features:
            if value is None or str(value).lower() in ("null", ""):
                final_data[feat]=0
            else:
                try:
                    final_data[feat] = int(float(value))
                except (ValueError, TypeError):
                    print(f'Warning: Could not convert {feat} with value {value} to integer. Setting to 0.')
                    final_data[feat] = 0
        else:
            if value is None or str(value).strip().lower() in ('null',''):
                final_data[feat]=None
            else:
                final_data[feat]=str(value).strip()
        
    return final_data

def extract_data_from_des(description:str):

    if not description or len(description)<10:
        print("Description too short.")
        return {}

    user_prompt = extraction_prompt(description)

    json_string = call_genai_api(user_prompt, GEMINI_KEY)

    structured_data = parse_and_validate_json(json_string)

    return structured_data     
