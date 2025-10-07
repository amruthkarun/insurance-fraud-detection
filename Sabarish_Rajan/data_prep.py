import pandas as pd
import numpy as np
from pandas import DataFrame


class DataPreprocessor:


    def __init__(self):
        pass

    def _clean(self, df:DataFrame):
        df = df.drop_duplicates()
        df = df.replace('?', np.nan)


        df['authorities_contacted'] = df['authorities_contacted'].replace(np.nan, 'No')
        df['fraud_reported'] = df['fraud_reported'].replace({'Y':1, 'N':0}).astype(int)
        df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
        df['property_damage'] = df['property_damage'].fillna('NO')
        df['police_report_available'] = df['police_report_available'].fillna('NO')


        df['injury_claim_ratio']=df['injury_claim']/df['total_claim_amount']
        df['property_claim_ratio']=df['property_claim']/df['total_claim_amount']
        df['vehicle_claim_ratio']=df['vehicle_claim']/df['total_claim_amount']

    #based on the correlation data we drop the following columns
        df = df.drop(['age','insured_hobbies','auto_make', 'policy_number','injury_claim','property_claim','vehicle_claim', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip', 'auto_model', 'auto_year'], axis=1)

        return df

    def _preprocessing(self, df:DataFrame):
    #Policy_annual premium has few outliers, we will remove them
        Q1 = df['policy_annual_premium'].quantile(0.25)
        Q3 = df['policy_annual_premium'].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df['policy_annual_premium'] >= Q1 - 1.5 * IQR) & (df['policy_annual_premium'] <= Q3 + 1.5 * IQR)
        df = df.loc[filter]

    # total_claim_amount has few outliers, we will remove them
        Q1 = df['total_claim_amount'].quantile(0.25)
        Q3 = df['total_claim_amount'].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df['total_claim_amount']>=Q1 -1.5*IQR) & (df['total_claim_amount']<=Q3 + 1.5*IQR)
        df = df.loc[filter]

        return df