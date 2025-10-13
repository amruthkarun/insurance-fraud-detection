import re
import pandas as pd
from imblearn.over_sampling import SMOTE


class ClaimPreprocessor:
    def __init__(self, file_path):
        """Initialize with file path"""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load dataset from CSV"""
        self.df = pd.read_csv(self.file_path)
        return self.df

    def range_to_mean(self, val):
        
        if pd.isna(val): return 0
        val = str(val).lower()
        if "none" in val: return 0
        if "new" in val: return 0
        if "no change" in val: return 0
        if "under" in val: return 0.5
        if "more than" in val:
            return int(re.findall(r"\d+", val)[0]) + 1
        nums = re.findall(r"\d+", val)
        if len(nums) == 1:
            return int(nums[0])
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2
        return 0

    def preprocess(self):
        df = self.df.copy()

        df = df.drop(columns=['PolicyNumber', 'RepNumber', 'Year'], errors="ignore")

        cols_to_convert = [
            'VehiclePrice','Days:Policy-Accident','Days:Policy-Claim',
            'PastNumberOfClaims','AgeOfVehicle','AgeOfPolicyHolder',
            'NumberOfSuppliments','AddressChange-Claim','NumberOfCars'
        ]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col].apply(self.range_to_mean)

        if 'FraudFound' in df.columns:
            df['FraudFound'] = df['FraudFound'].map({'Yes':1, 'No':0})

        self.df = df
        return self.df



class SMOTEHandler:
    def __init__(self, random_state=42):
        
        self.smote = SMOTE(random_state=random_state)
        self.X_resampled = None
        self.y_resampled = None

    def fit_resample(self, X, y):
        
        self.X_resampled, self.y_resampled = self.smote.fit_resample(X, y)
        return self.X_resampled, self.y_resampled

    def get_resampled_data(self):
        
        return self.X_resampled, self.y_resampled






    


