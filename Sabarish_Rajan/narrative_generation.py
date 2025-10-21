import shap
from google import genai
from google.genai.errors import APIError
import pandas as pd

class GenerateNarrative:

    def __init__(self, model_, gemini_client_):
        self.model = model_
        self.gemini_client = gemini_client_

        try:
            self.explainer = shap.TreeExplainer(self.model.named_steps["classifier"])
            print("SHAP Explainer Initialized")
            self.expected_value = self.explainer.expected_value
        except Exception as e:
            print(f"Error initializing explainer:{e}")
            self.explainer = None
            self.expected_value = 0.0
        
        
        
    def generate_narrative(self, risk_level: str, probability: float, raw_data: dict, processed_df : pd.DataFrame):

        if not self.gemini_client:
            return "GenAI services not available"
    
        if self.explainer and risk_level in ["Low Risk", "Medium Risk", "High Risk"]:
            scaled_df_for_shap = self.model.named_steps["scaler"].transform(processed_df)
            shap_value = self.explainer.shap_values(scaled_df_for_shap)[0]
            feature_names = processed_df.columns

            shap_series = pd.Series(shap_value, index=feature_names)

            top_drivers_abs = shap_series.abs().sort_values(ascending=False).head(5)
            top_drivers_with_sign = shap_series.loc[top_drivers_abs.index]
            self.shap_values = shap_value
            self.feature_names = feature_names
        prompt_lines = [
            "You are Fraud Analyst AI. Generate a concise, professional justification for the prediction of this insurance claim.",
            f"The predictor model returned a **{risk_level}** status with a probability of **{probability:.1%}** of fraud.",
            "Analyze the top 5 most influential factors and their actual values to explain why the model flagged the claim.",
            "The influential factors are:"
        ]

        for feature, shap_value in top_drivers_with_sign.items():

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
            +"Check the Accident state and the the accident city."
        )    

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=final_prompt
            )
            return response.text
        except APIError as e:
            return "API Error: Could not generate narrative."
        except Exception as e:
            print(f"Error occured:{e}")
            return "Unexpected error encountered. Cannot generate narrative."
    