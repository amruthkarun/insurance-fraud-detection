from google import genai
from dotenv import load_dotenv
#from google.genai.errors import APIError
from google.genai.types import Tool, GenerateContentConfig
import json
from typing import Dict,Any
import requests
import time
import os

load_dotenv()

class extract_incident_data():
    def __init__(self):
        self.GEMINI_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        else:
            try:
                self.gemini_client = genai.Client(api_key = self.GEMINI_KEY)
                print('GenAI Client loaded successfully😊')
            except Exception as e:
                print(f'Error loading Client: {e}')
        
         
    required_features = [
    'incident_date',
    'incident_type',
    'collision_type',
    'incident_severity',
    'authorities_contacted',
    'incident_state',
    'incident_city',
    'incident_location',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'property_damage',
    'bodily_injuries',
    'witnesses',
    'police_report_available',
    'total_claim_amount',
    'injury_claim',
    'property_claim',
    'vehicle_claim',
    'auto_make',
    'auto_model',
    'auto_year'
    ]

    numeric_features = [
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'total_claim_amount',
    'injury_claim',
    'property_claim',
    'vehicle_claim',
    'auto_year'
    ]
    
    def extraction_prompt(self, description:str):

        self.prompt=f"""
    You are an expert insurance claims processor and data extraction agent.
    Your sole purpose is to analyse the following incident description provided by the user and 
    extract all the relevant details into a clean JSON object.
    
    **INSTRUCTIONS**
    1. Extract data for the following features {",".join(self.required_features)}.
    2. Ensure numeric features (like 'total_claim_amount') are clean integers/floats.
    3. If any datapoint is not explicitly mentioned, you MUST set its value to 0(for numbers) or 'null'. Do NOT guess or hallucinate values.
    4. Return only JSON objects. Do not include any conversational texts outside of the JSON block itself.
    
    **USER DESCRIPTION**
    
    {description}.
    
    ---
    """

        return self.prompt

    def schema_gen(self):
        all_feat = self.required_features
        num_feat = self.numeric_features
        properties={}
        for feat in all_feat:
            prop_type = 'integer' if feat in num_feat else 'string'

            properties[feat]={"type":prop_type}

        return{
            "type":"object",
            'properties':properties,
            'property_ordering':all_feat
        }


    def call_genai_api(self,prompt:str):
        model_name = 'gemini-2.5-flash-preview-05-20'
        max_tries = 5
        response_schema = self.schema_gen()
        response_schema['required'] = self.required_features
        '''tools_to_use = [
            types.Tool(google_search={})
        ]'''
        json_string =""

        for tries in range(max_tries):
            try:
                response = self.gemini_client.models.generate_content(
                    model = model_name,
                    contents = self.prompt,
                    #tools= tools_to_use,
                    config = GenerateContentConfig(
                        response_schema = response_schema,
                        response_mime_type = 'application/json'
                    )
                )

                json_string = response.text.strip()
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

    def parse_and_validate_json(self, json_string:str):

        data = {}

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM:{e}")
            return {}
    
        final_data = {}

        for feat in self.required_features:
            value = data.get(feat)
        
            if feat in self.numeric_features:
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

    def extract_data_from_des(self, description:str):

        if not description or len(description)<10:
            print("Description too short.")
            return {}

        try:
            user_prompt = self.extraction_prompt(description)

            json_string = self.call_genai_api(user_prompt)

            structured_data = self.parse_and_validate_json(json_string)
        except Exception as e:
            print(f'Error:{e}')
            return f'Error:{e}'

        return structured_data 

    


    

