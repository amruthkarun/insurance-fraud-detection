import tensorflow as tf
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from keras_tuner import RandomSearch

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import F1Score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sqlalchemy import create_engine
import psycopg2
import random

#For code reproducibility

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


#connecting to PostgreSQL database
DB_URL = 'postgresql://postgres:1905@localhost:5432/Insuarance_fraud'
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


#based on the correlation data we drop the following columns
df = df.drop(['age','insured_hobbies','auto_make', 'policy_number','injury_claim','property_claim','vehicle_claim', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip', 'auto_model', 'auto_year'], axis=1)

df_corr = df[df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index].corr()

'''plt.pyplot.figure(figsize=(12,10))
sns.heatmap(df_corr, annot=True)
plt.pyplot.show()
'''
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
x_train, X_test, y_train_1,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify = y)
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train_1, test_size = 0.2, random_state = 42, stratify = y_train_1)
#X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

#print(X_res.shape)
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()
#Model Building

'''earlyStopper = EarlyStopping(
    monitor = 'val_f1_score',
    patience = 20,
    mode = 'max',
    restore_best_weights = True
)'''

# Class_weights

total_samples = len(y_train)

n_fraud = sum(y_train)
n_nfraud = total_samples - n_fraud

weight_for_0 = (1 / n_nfraud) * (total_samples / 2.0)
weight_for_1 = (1 / n_fraud) * (total_samples / 2.0)

class_weights = {0: weight_for_0, 1: weight_for_1}

#Code not complete yet🥲

#Building a Neural Network Model
def build_model(hp):
    model = Sequential()
    model.add(tf.keras.Input(shape=X_res_scaled.shape[1:]))

    hp_regularizer_choice = hp.Choice('regularizer_type', values = ['l1', 'l2', 'l1_l2', 'none'])
    hp_regularizer_rate = hp.Float('regularizer_rate', min_value = 1e-5, max_value = 1e-2, sampling = 'log')

    if hp_regularizer_choice == 'l1':
        regularizer = regularizers.L1(hp_regularizer_rate)
    elif hp_regularizer_choice == 'l2':
        regularizer = regularizers.L2(hp_regularizer_rate)
    elif hp_regularizer_choice == 'l1_l2':
        regularizer = regularizers.L1L2(l1 = hp_regularizer_rate, l2 = hp_regularizer_rate)
    else:
        regularizer = None

    for i in range(hp.Int('num_hidden_layers', min_value = 1, max_value = 5 )):
        model.add(Dense(units = hp.Int('units_'+str(i), min_value = 64, max_value = 512, step = 32),
            activation = 'relu',
            kernel_initializer = 'he_normal',
            kernel_regularizer=regularizer))
        model.add(Dropout(rate=hp.Float('dropuout_'+str(i), min_value = 0.2, max_value = 0.5, step = 0.1)))


    model.add(Dense(1, activation = 'sigmoid'))

    learning_rate = hp.Choice('learning_rate_', values = [1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(loss = 'BinaryCrossentropy',
        optimizer = RMSprop(learning_rate = learning_rate),
        metrics = [F1Score(average = 'weighted', name = 'f1_score')]
    )


    return model

tuner = RandomSearch(
    build_model,
    objective = 'val_f1_score',
    max_trials = 30,
    executions_per_trial = 1,
    project_name = 'fraud_det'
)



tuner.search(X_res_scaled, y_train,epochs = 100, validation_data=(X_valid_scaled, y_valid), class_weight = class_weights)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.get_best_models(num_models = 1)[0]

y_prob = best_model.predict(X_test_scaled)

thresholds = np.arange(0.1, 0.9, 0.05)

f1_scores = [f1_score(y_test, (y_prob>t).astype(int)) for t in thresholds]

best_threshold = thresholds[np.argmax(f1_scores)]
print('Best Threshold:',best_threshold)
y_pred = (y_prob >= best_threshold).astype(int)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

end_time = time.time()

print('Time taken for model training and prediction: ', round(end_time - start_time,2), 'seconds')

