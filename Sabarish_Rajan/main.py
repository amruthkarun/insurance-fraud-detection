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

print(df[df['authorities_contacted']=='NaN']['fraud_reported'])