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
from data_prep import DataPreprocessor

#connecting to PostgreSQL database
DB_URL = 'postgresql://postgres:1905@localhost:5432/Insuarance_fraud'
engine = create_engine(DB_URL)
query = "SELECT * FROM claims"
df = pd.read_sql(query, engine)


processor = DataPreprocessor()

df = processor._clean(df)

#print(df.info())

#Encoding categorical data
cat_cols = df.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = pd.DataFrame(ohe.fit_transform(df[cat_cols]), columns = ohe.get_feature_names_out(cat_cols))

joblib.dump(ohe, 'OneHotEncoder.pkl')

df = pd.concat([df, encoded_cols],axis=1)
df = df.drop(cat_cols, axis=1)
#print(df.shape)

df = processor._preprocessing(df)

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
X = X.fillna(0)
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