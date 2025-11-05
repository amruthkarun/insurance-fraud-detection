# Insurance Fraud Detection

This project deals with building a Machine Learning model that detects if an Insurance claim is fraud or not. Why is that important? Simple, insurance fraud accounts for almost $6 billion(Rs. 529 billion) losses for the insurance industry in India, that is about 8.5% of the total industry revenue. This directly leads to an increase in the annual premium payment of an honest insurance policy holders.

This project implements a Machine Learning model to detect fraudulent insurance claims and deploys it as a real-time service. The primary objective is to build an accurate, robust, and interpretable system capable of identifying suspicious activities, allowing insurance companies to minimize payouts on fraudulent claims while ensuring legitimate claims are processed efficiently.

Due to the heavily imbalanced nature of fraud datasets, the final XGBoost model utilizes techniques like SMOTE oversampling and a custom probability threshold optimized for the F1-Score.

## 1. Problem Statement

**Date - 11th September 2025**

Develop a platform using Python to analyze and detect fraudulent insurance claims using open-source datasets and historical claim data. Build a web application with FastAPI, apply machine learning models for fraud detection, and assess risk levels. Implement threading for efficient data processing and integrate PostgreSQL for data management. (e.g., improvisations can be made in feature engineering, incorporating external risk factors, and applying responsible AI concepts for ethical decision-making.)

### Objectives:

Collect, preprocess, and enrich insurance claim data using Python libraries and APIs.
Apply machine learning models (e.g., Logistic Regression, Random Forest) for fraud detection and risk assessment.
Implement feature engineering techniques to enhance model accuracy and reliability.
Develop a web application with FastAPI, featuring interactive dashboards using Plotly or Bokeh.
Integrate PostgreSQL for storing claim data and risk assessments, utilizing threading for efficient data processing tasks.

### GenAI Objective: Use GenAI to extract and process information from unstructured data sources.

## 2. Dataset

**Source**: https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection<br>
**Description**: The dataset used for this analysis contains various features related to insurance claims, including policy details, customer information, accident specifics, and the target variable indicating whether a claim is fraudulent or not.

## 3. Features

- **Web Interface:** A simple HTML/JavaScript frontend to submit new insurance claims.
- **NLP Parsing:** Automatically extracts structured features (e.g., policy number, incident type) from an unstructured text description of the incident.
- **ML-Powered Prediction:** Uses an XGBoost model to classify each claim as **"Genuine"** or **"Fraudulent"**.
- **Explainable AI (XAI):** Generates SHAP (SHapley Additive exPlanations) values to understand _why_ the model made a specific prediction.
- **Generative AI Narrative:** Uses the Gemini 2.5 Flash model to translate complex SHAP values into a simple, human-readable summary.
- **Data Persistence:** All submitted claims and their predictions are stored in a PostgreSQL database.

### Tech Stack

- **Backend:** **FastAPI** (Python)
- **Frontend:** **HTML**, **JavaScript** (served as templates by FastAPI)
- **Database:** **PostgreSQL**
- **Machine Learning:** **Scikit-learn** (for `OneHotEncoder`), **XGBoost**
- **Explainability:** **SHAP**
- **Generative AI:** **Google Gemini 2.5 Flash Preview**
- **Containerization:** **Docker**, **Docker Compose**
- **Orchestration:** **Kubernetes (K8s)**

## 4. Application Flow

Here is the step-by-step data flow when a user submits a claim:

1.  **Submission:** A user fills out a form in the web UI, providing a text description of the incident.
2.  **API Endpoint:** The form data is sent to a `POST` endpoint in the **FastAPI** backend.
3.  **NLP Parsing:** The `nlp_parser.py` module extracts key features from the raw text.
4.  **Database Insert:** The `insert_to_database.py` module logs the incoming claim data to the **PostgreSQL** database.
5.  **Preprocessing:** The `preprocessor.py` module uses the saved `OneHotEncoder` to prepare the data for the model.
6.  **Prediction:** The preprocessed data is fed into the loaded **XGBoost model** (`.pkl` file), which outputs a fraud probability.
7.  **Explanation (XAI):** **SHAP** values are calculated to identify the top factors contributing to the model's decision.
8.  **Narrative Generation:** The `narrative_generation.py` module sends these SHAP values to the **Gemini API** to generate a simple English explanation.
9.  **Response:** The FastAPI backend returns the final prediction (e.g., "Fraud"), the probability score, and the AI-generated narrative to the user in the frontend.

## 5. Getting Started

You can run this project using Docker Compose (recommended for local testing) or Kubernetes (for a full deployment).

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker](https://www.docker.com/products/docker-desktop/) & [Docker Compose](https://docs.docker.com/compose/install/)
- (For K8s) [kubectl](https://kubernetes.io/docs/tasks/tools/)
- (For K8s) [Minikube](https://minikube.sigs.k8s.io/docs/start/) or a cloud-based Kubernetes cluster.
- A **Gemini API Key**.

### 1. Using Docker Compose (Recommended)

1.  **Clone the repository:**

    ```sh
    git clone [your-repository-url]
    cd [repository-name]
    ```

2.  **Create an environment file:**
    Create a file named `.env` in the root directory and add your credentials.

    ```.env
    # PostgreSQL Credentials
    POSTGRES_USER=myuser
    POSTGRES_PASSWORD=mypassword
    POSTGRES_DB=fraud_db
    POSTGRES_HOST=db

    # Gemini API Key
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
    ```

    _Note: The `docker-compose.yml` file is configured to use these variables._

3.  **Build and run the containers:**

    ```sh
    docker-compose up -d --build
    ```

    This will build the `fastapi-app` image and pull the `postgres` image. The `-d` flag runs them in detached mode.

4.  **Access the application:**
    Open your browser and navigate to **`http://localhost:8000`**.

### 2. Using Kubernetes

1.  **Ensure your `kubectl` is configured** to point to your cluster (e.g., `minikube start`).

2.  **Create a secret for your API key:**

    ```sh
    kubectl create secret generic gemini-api-key --from-literal=API_KEY=YOUR_GEMINI_API_KEY_HERE
    ```

    _(You will also need to create secrets for your database credentials or manage them with a `ConfigMap` and `StatefulSet` for production. The provided files assume this is set up)._

3.  **Apply the Kubernetes configuration files:**
    Apply the files in the correct order (database first, then the application).

    ```sh
    # Start the PostgreSQL Database
    kubectl apply -f db-deployment.yaml
    kubectl apply -f db-service.yaml

    # Wait for the database to be fully running

    # Deploy the FastAPI Application
    kubectl apply -f deployment.yaml
    kubectl apply -f app-service.yaml
    ```

4.  **Find your service URL:**
    If you are using Minikube, you can expose the service:
    ```sh
    minikube service fraud-detection-service
    ```
    This will open the application in your browser.

## 6. Project Structure

A high-level overview of the key files and directories:

- **Sabaris_Rajan/** (Project Root Directory)

  - `app/` (FastAPI Application & Web Assets)

    - `static/`
    - `templates/`
      - `index.html` (Frontend)
    - `__pycache__/` (Ignorable Cache Folder)
    - `.env` (Local Environment Variables)
    - `app-service.yaml` (Kubernetes Service for FastAPI)
    - `data_prep.py` (Data cleaning and feature engineering module)
    - `data_preprocessor.pkl` (Fitted OneHotEncoder/Preprocessor Artifact)
    - `db-deployment.yaml` (Kubernetes Deployment for PostgreSQL)
    - `db-service.yaml` (Kubernetes Service for PostgreSQL)
    - `deployment.yaml` (Kubernetes Deployment for FastAPI)
    - `dockerfile` (Build blueprint for the FastAPI app)
    - `Feature_order.pkl` (Artifact storing the expected feature list/order)
    - `insert_to_database.py` (Module for DB insertion logic)
    - `ML_Model.pkl` (Serialized XGBoost Model Artifact)
    - `Model.py` (Likely the training script for the ML model)
    - `my_API_2.py` (Main FastAPI application entry point)
    - `narrative_generation.py` (Module for Gemini API calls)
    - `nlp_parser.py` (Module to parse claim descriptions)
    - `preprocessor.py` (Module for OneHotEncoding logic)
    - `requirements.txt`

  - `README.md`
  - `Developer_Documentation.md`
  - `data/`
    - `Fraud_detection.csv` (Source Data File)
  - `docker-compose.yml` (Docker Compose file for local development)
  - `.dockerignore`

## 7. Model Performance and Results

| Model                              | Precision | Recall (Sensitivity) | F1-Score |
| :--------------------------------- | :-------- | :------------------- | :------- |
| **Logistic Regression (Baseline)** | 0.46      | 0.70                 | 0.55     |
| **Descision Tree Classifier**      | 0.46      | 0.70                 | 0.55     |
| **Random Forest Classifier**       | 0.80      | 0.51                 | 0.61     |
| **XGBoost Classifier**             | 0.64      | 0.82                 | 0.72     |
| **XGBoost Classifier**             | 0.64      | 0.82                 | 0.72     |
| **Neural Networks**                | 0.64      | 0.71                 | 0.67     |

Key Findings : The XGBoost Classifier was identified as the best performing model. By using SMOTE oversampling and optimizing the final prediction threshold for the F1-Score, the model achieved an F1-Score of 0.72 on the test set. The high Recall (estimated around 0.82) indicates the model is highly effective at minimizing missed fraud cases (False Negatives), which is a critical priority in insurance fraud detection. The final model (ML_Model.pkl) and OneHotEncoder.pkl are saved for deployment.
