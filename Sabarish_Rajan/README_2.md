# Insurance Fraud Detection

This project deals with building a Machine Learning model that detects if an Insurance claim is fraud or not. Why is that important? Simple, insurance fraud accounts for almost $6 billion(Rs. 529 billion) losses for the insurance industry in India, that is about 8.5% of the total industry revenue. This directly leads to an increase in the annual premium payment of an honest insurance policy holders. 

This project deals with building a model that uses GenAI and Machine Learning concepts to predict if an insurance claim is fraud or not. 

(This file has a detailed description of what are the steps I took to complete this project)

## Problem Statement 
#### Date - 11th September 2025
Develop a platform using Python to analyze and detect fraudulent insurance claims using open-source datasets and historical claim data. Build a web application with FastAPI, apply machine learning models for fraud detection, and assess risk levels. Implement threading for efficient data processing and integrate PostgreSQL for data management. (e.g., improvisations can be made in feature engineering, incorporating external risk factors, and applying responsible AI concepts for ethical decision-making.)

### Objectives:

Collect, preprocess, and enrich insurance claim data using Python libraries and APIs.
Apply machine learning models (e.g., Logistic Regression, Random Forest) for fraud detection and risk assessment.
Implement feature engineering techniques to enhance model accuracy and reliability.
Develop a web application with FastAPI, featuring interactive dashboards using Plotly or Bokeh.
Integrate PostgreSQL for storing claim data and risk assessments, utilizing threading for efficient data processing tasks.
### GenAI Objective: Use GenAI to extract and process information from unstructured data sources.

## Project Development Log

#### I have decided to use a parallel method of completing this project. I will be working on building the model and frontend/ backend development parallely. 

### Date - 12th September 2025

#### Model

- Forked the repository and will be making changes there before pushing it to the main branch.
- Found the dataset from Kaggle that has 39 feature and 1000 data values (link - https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection)
- Started cleaning the data. Removed duplicates, replaced missing and non-sensical values. Imputed a few data points with the mean, etc.
- Once I was satisfied with the dataset, I performed a basic Exploratoty Data analysis on it.   
    - I plotted the distribution of each feature to get which datasets to normalize.
    - Made a correlation heatmap to figure out features with highest correlation and removed one of these features.
    - Made a box plot to identify features with outliers.
    - Performed Univariate analysis of each feature.
- After the EDA, I again cleaned the data, this time by removing some unnecessary features and adding some combined features like 'claim_to_premium_ratio' and 'severity_of_incident'.

#### Frontend/ Backend

- Watched some tutorials on FastAPI and web framework building.
- Followed a tutorial and built a very basic API (not related to the project) to get the hang of the API building process.

### Date - 15th September 2025
#### Model

- Completed EDA and feature Engineering. 
- Started building a basic pipeline using Random Forest Classifier.
- Used Grid Search CV to get the best estimator and tested the best model. 
- F1 score = 57, Accuracy = 64%
- Switched Random Forest with XGBoost Classifier and again used Grid Search CV to find the best estimator.
- F1 score = 69, Accuracy = 80%
- Used SQLAlchemy to take data directly from PostgreSQL database(pg Admin 4).

#### Frontend/ Backend
- Built a preliminary API that is successfully able to take in data from the user. (Used Gemini to understand some HTML and JavaScript concepts to write the index.html file)
- Tried to get the 'post' part of my_API file to work as well, but there is some problem with connecting to the pickle file.

#### GenAI
- Watched tutorials on Transformer models and LLM's.

### Date - 18th September 2025

#### Model 
- I got the F1 score to 70 and accuracy upto 83% by tunning some hyperparameters.
- To get a better model, tried usinf RandomizedSearchCV, but that just made things worse. F1 score dropped to 54.
- Tried adding more hyperparameters and various values into Grid Search but the results were the same, F1 score = 70, Accuracy = 83%
- Switched the model to a Stacking Classifier and used Logistic Regression, KNN and Decision tree as base models and XGBoost as meta_classifier. F1 score = 54, Accuracy = 79%.
- Keeping this model intact will be building another model based on Neural Networks and check that out.

#### Frontend/ Backend
- Finally was able to solve the data validation issue and got the predictions to be displayed on the framework.
- Used style block in the head of index.html to beautify the API a little.

### Date - 22nd September 2025

#### Model

- Made a new model based on Neural Networks. Made a very simple model with 5 layers(4 hidden 1 output).
- The first model had a F1 score of 0.47 so used better optimizers and initializers.
- The max F1 score after multilple changes was the same as XGBoost, 0.70, with an accuracy of 83%.
- Currently using keras-tuner to find the best hyperparameters for the model. Will be working on that end next as well.
