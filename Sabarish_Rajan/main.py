import tensorflow as tf
import matplotlib as plt
import pandas as pd
import numpy as np

df = pd.read_excel('Insuarance_fraud_claim.xlsx')

#print(df.head())

y = df['fraud_reported']

#print(y.head())

df = df.drop(columns=['fraud_reported']).copy()
#print(df.head())

is_nan = df['authorities_contacted'] == 'NaN'
id_nan_df = df[is_nan]
print(is_nan_df['fraud_reported'])