#import tensorflow as tf
import joblib
import time as time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
'''import plotly.express as px
import plotly.graph_objects as go'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sqlalchemy import create_engine
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

#connecting to PostgreSQL database
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)
query = "SELECT * FROM claims"
df = pd.read_sql(query, engine)


df = df.drop_duplicates()
df = df.replace('?', np.nan)


df['authorities_contacted'] = df['authorities_contacted'].replace(np.nan, 'No')
df['fraud_reported'] = df['fraud_reported'].replace({'Y':1, 'N':0}).astype(int)
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna('NO')
df['police_report_available'] = df['police_report_available'].fillna('NO')

df_corr = df[df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index].corr()

'''plt.pyplot.figure(figsize=(12,10))
sns.heatmap(df_corr, annot=True)
plt.pyplot.show()'''

df['injury_claim_ratio']=df['injury_claim']/df['total_claim_amount']
df['property_claim_ratio']=df['property_claim']/df['total_claim_amount']
df['vehicle_claim_ratio']=df['vehicle_claim']/df['total_claim_amount']

#based on the correlation data we drop the following columns
df = df.drop(['age','insured_hobbies','auto_make', 'policy_number','injury_claim','property_claim','vehicle_claim', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip', 'auto_model', 'auto_year'], axis=1)

df_corr = df[df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index].corr()

'''plt.figure(figsize=(12,10))
sns.heatmap(df_corr, annot=True)
plt.show()'''

#print(df.info())

#Encoding categorical data
cat_cols = df.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = pd.DataFrame(ohe.fit_transform(df[cat_cols]), columns = ohe.get_feature_names_out(cat_cols))

joblib.dump(ohe, 'OneHotEncoder.pkl')

df = pd.concat([df, encoded_cols],axis=1)
df = df.drop(cat_cols, axis=1)
#print(df.shape)

#identifying outliers
'''num_cols = df.select_dtypes(include=['int64','float64']).columns
for cols in num_cols:
    plt.pyplot.figure()
    sns.boxplot(x = df[cols])
    plt.pyplot.show()'''

#Policy_annual premium has few outliers, we will remove them
Q1 = df['policy_annual_premium'].quantile(0.25)
Q3 = df['policy_annual_premium'].quantile(0.75)
IQR = Q3 - Q1
filter = (df['policy_annual_premium'] >= Q1 - 1.5 * IQR) & (df['policy_annual_premium'] <= Q3 + 1.5 * IQR)
df = df.loc[filter]

#Umbrella limit has 50% of the data as 0, hence we create a new binary column
#df['umbrella_limit_'] = np.where(df['umbrella_limit']==0,0,1)
#df= df.drop('umbrella_limit', axis=1)
#Making an umbrella limit column did not improve the model score, hence we are not using it.


# total_claim_amount has few outliers, we will remove them
Q1 = df['total_claim_amount'].quantile(0.25)
Q3 = df['total_claim_amount'].quantile(0.75)
IQR = Q3 - Q1
filter = (df['total_claim_amount']>=Q1 -1.5*IQR) & (df['total_claim_amount']<=Q3 + 1.5*IQR)
df = df.loc[filter]
'''col = ['policy_annual_premium', 'total_claim_amount']
for cols in df[col]:
    plt.pyplot.figure()
    sns.boxplot(x = df[cols])
    plt.pyplot.show()'''

#getting the number of fraud and non fraud cases
class_counts = df['fraud_reported'].value_counts()
fraud = class_counts.loc[1.0]

total_claims = len(df)

print('%age of fraud classes :', round(fraud/total_claims * 100,2))
print('% of non fraud classes :', round((total_claims - fraud)/total_claims * 100,2))


#Splitting the data into X and y
X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']
#feature Engineering

X['claim_to_premium_ratio'] = X['total_claim_amount']/(X['policy_annual_premium']+0.01)
X['severity_of_incident']=X['total_claim_amount']/(X['witnesses']+0.01)


#Splitting the data into train and test
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify = y)

X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

start_time = time.time()
#Model Building
full_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(random_state = 42))#XGBClassifier
])

imbalance_ratio = (len(y_res)-sum(y_res))/sum(y_res)

param_grid = {
    'classifier__n_estimators': [200, 400, 600],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__scale_pos_weight': [imbalance_ratio]
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    #n_iter=100,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring='f1',
    refit='f1'
)

grid_search.fit(X_res, y_res)

print(grid_search.best_params_)

'''cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res = cv_res.sort_values(by='rank_test_score', ascending=True)
print(cv_res[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])'''

best_model = grid_search.best_estimator_

joblib.dump(best_model, 'ML_Model.pkl')

y_prob = best_model.predict_proba(X_test)

thresholds = np.arange(0.1, 0.9, 0.05)

f1_scores = [f1_score(y_test, (y_prob[:,1]>t).astype(int)) for t in thresholds]

best_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_prob[:,1] >= best_threshold).astype(int)


print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

end_time = time.time()

print('Time taken for model training and prediction: ', round(end_time - start_time,2), 'seconds')

#The F1 score is 0.70 which is good considering the data is highly imbalanced. 
# But using feature importance did not improve the score, hence we are not using it.