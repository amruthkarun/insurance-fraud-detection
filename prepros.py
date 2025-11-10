#importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

ins_df=pd.read_csv('D:\python\insurance fraud\insurance_claims.csv')

print(ins_df.head())
print(ins_df.shape)
ins_df.info()

# checking for missing values
print(ins_df.isnull().sum())
print(ins_df.value_counts(['police_report_available','bodily_injuries','property_damage']))


#cleaning missing values 
ins_df=ins_df.drop(columns=['_c39'],axis=1)
ins_df=ins_df.dropna()
print(ins_df.head())

#EDA
plt.figure(figsize=(7,7))
sns.countplot('fraud_reported')
plt.show()

le=LabelEncoder()

categorical_data=ins_df.select_dtypes(include=['object']).columns
for col in categorical_data:
    ins_df[col]=le.fit_transform(ins_df[col])

#heatmap

plt.figure(figsize = (18, 12))

corr = ins_df.corr()

sns.heatmap(data = corr, annot = True)
plt.show()

print(ins_df[:5])
print(ins_df.nunique())


#saving the preprocessed file
ins_df.to_csv("processed_test.csv", index=False)