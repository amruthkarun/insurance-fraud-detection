#importing libs 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

df=pd.read_csv('D:\python\insurance fraud\processed.csv')

#splitting fro x & y

x=df.drop(columns=['age','policy_number','policy_bind_date','policy_state','policy_csl','umbrella_limit','insured_zip','insured_relationship','incident_date','incident_state','incident_city','incident_location','vehicle_claim','auto_make','fraud_reported'],axis=1)
y=df['fraud_reported']

print(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.1)

print(x_train.shape,x_test.shape)

#checking fro cross-val scores of diff models

print('rf',cross_val_score(RandomForestClassifier(n_estimators=60),x,y,cv=10))

print('KNN',cross_val_score(KNeighborsClassifier(),x,y,cv=10))

print('SVC',cross_val_score(SVC(),x,y,cv=10))

print('XGB',cross_val_score(XGBClassifier(),x,y,cv=10))

# xgb model

model=XGBClassifier( n_estimators=250,
    learning_rate=0.25,
    max_depth=4)

#cross validating the data set for limiting data over fitting
cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print("accuracy for training:",training_data_accuracy)

#testdata scores for various evaluation metries
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print('accuracy on test data:',test_data_accuracy)
test_precision = precision_score(y_test,x_test_prediction)
print('precision on the test data',test_precision)
recall_test = recall_score(y_test,x_test_prediction)
print('recall on the test data',recall_test)
f1_test = f1_score(y_test,x_test_prediction)
print('f1 score on the test data',f1_test)
conf_matrix_test = confusion_matrix(y_test,x_test_prediction)
print('confusion mtx:',conf_matrix_test)

import os
#saving the model
save_path = r'D:\python\insurance fraud\model.pkl'
joblib.dump(model, save_path)
print("Model saved successfully at:", os.path.abspath(save_path))


#saving test_data
x_test.to_csv('test.csv',index=False)
