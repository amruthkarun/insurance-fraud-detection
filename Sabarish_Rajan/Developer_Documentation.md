# 🧑‍💻 Developer Documentation

This document provides a deep-dive into the technical architecture, module responsibilities, and setup procedures for the **End-to-End Insurance Fraud Detection** project.

---

## 1. Local Development Setup (Without Docker)

While Docker is recommended, you can run the application locally for debugging.

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/SabarishRajan14/insurance-fraud-detection.git
    cd [repository-name]
    ```

2.  **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up the database:**
    You must have a PostgreSQL server running locally or remotely.

5.  **Set environment variables:**
    Create a `.env` file in the root directory. The application uses `python-dotenv` to load these.

    ```.env
    # Set your local or remote DB connection string
    DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/fraud_db

    # Gemini API Key
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
    ```

6.  **Run the database migrations/init script:**

7.  **Run the FastAPI server:**
    The application object `app` is in `my_API_2.py`.
    ```sh
    uvicorn app.my_API_2:app --reload
    ```
    The `--reload` flag will automatically restart the server on code changes. The app will be available at `http://127.0.0.1:8000`.

---

## 2. 📦 Module Breakdown

The application is structured in a modular way. Here are the core components:

### `app/my_API_2.py`

- **Role:** The main **FastAPI application**. This file orchestrates the entire process.
- **Key Functions:**
  - Initializes the FastAPI `app`.
  - Loads the XGBoost model and encoder from `.pkl` file on startup.
  - Defines the API endpoints.
  - **`@app.get("/")`**: Serves the `index.html` template.
  - **`@app.post("/predict")`** (or similar): This is the core logic endpoint.
    1.  Receives data from the form (as a Pydantic model).
    2.  Calls `nlp_parser.py` to process the text description.
    3.  Calls `insert_to_database.py` to log the request.
    4.  Calls `preprocessor.py` to encode the features.
    5.  Runs the data through the loaded `model.predict_proba()`.
    6.  Calculates SHAP values.
    7.  Calls `narrative_generation.py` with the SHAP values.
    8.  Returns a JSON response with the prediction and narrative.

### `app/data_prep.py`

- **Role:** Data cleaning, preparation, and feature engineering.
- **Context:** This module is used in two places:
  1.  **During Training:** In `ML_Model.py` (the training script) to prepare the training data.
  2.  **During Inference:** In `my_API_2.py` to apply the _exact same_ transformations to live, incoming data before prediction.

### `app/preprocessor.py`

- **Role:** Handles categorical feature encoding.
- **Context:** This module applies the `OneHotEncoder` (or other scalers/encoders) that was fitted during training.
- **Key Functions:**
  - The model `.pkl` file likely contains a `dict` or `tuple` such as `{'model': xgb_model, 'encoder': ohe_encoder}`.
  - This module contains the function that takes the raw feature-engineered `DataFrame`, applies the loaded `encoder.transform()`, and returns the model-ready array.
- **Warning:** Any changes to features in `data_prep.py` _must_ be reflected here and in the fitted `OneHotEncoder`. The model must be retrained if the feature set changes.

### `app/nlp_parser.py`

- **Role:** Extracts structured data from an unstructured "incident description" string.
- **Context:** This is a critical component for usability. Instead of forcing the user to fill 20 fields, they can write a paragraph.
- **Implementation:** Uses Gemini 2.5 Flash to get the feature values as JSON format.

### `app/insert_to_database.py`

- **Role:** Handles all database `INSERT` operations.
- **Context:** Called from `my_API_2.py` to log every incoming claim _before_ prediction. This ensures data is saved even if a downstream step (like the Gemini API) fails.
- **Key Functions:**
  - `insert_claim_data(claim_data: pd.DataFrame)`: Takes the data and writes it to the PostgreSQL database using `SQLAlchemy`.

### `app/narrative_generation.py`

- **Role:** Generates the human-readable XAI summary.
- **Context:** Takes the quantitative output from SHAP (e.g., `{"feature_A": 0.45, "feature_B": -0.2}`) and makes it understandable.
- **Key Functions:**
  - `get_narrative(shap_values: dict) -> str`:
    1.  Formats the SHAP values into a string.
    2.  Constructs a specific prompt for the Gemini 2.5 Flash model.
    3.  **Prompt Example:** `"You are an insurance adjuster. A claim was flagged as high-risk. Explain why in one simple sentence based on these factors: [shap_values_string]. Focus on the top 2-3 most important factors."`
    4.  Calls the Gemini API and returns the text response.

---

## 3. 🤖 Machine Learning Model (`ML_Model.py`)

- **This file is a _training script_, not part of the live application.**
- **Workflow:**

  1.  Load the raw training dataset (e.g., `fraud_data.csv`).
  2.  Run the dataset through `data_prep.py`.
  3.  Instantiate and `fit` the `OneHotEncoder` from `preprocessor.py`.
  4.  Train the `XGBoostClassifier`.
  5.  Save the _fitted model_ and the _fitted encoder_ to `model/xgboost_model.pkl` using `pickle`.

      ```python
      # Example of saving
      import pickle

      ohe = OneHotEncoder(...)
      #... fit ohe ...

      xgb = XGBClassifier(...)
      #... fit xgb ...

      model_payload = {
          'model': xgb,
          'encoder': ohe
      }

      with open("model/xgboost_model.pkl", "wb") as f:
          pickle.dump(model_payload, f)
      ```

---

## 4. 🗃️ Database Schema

- **Database:** PostgreSQL
- **Key Table(s):** `claims`

  ```sql
  -- Example Schema:
  CREATE TABLE claims (
      incident_date TIMESTAMP,
      policy_number VARCHAR(100),
      -- ... other features extracted by nlp_parser ...

      is_fraud_prediction INTEGER, -- 0 or 1
  );
  ```

---

## 5. 🐳 Containerization & Deployment

### Docker

- **`Dockerfile`**: This is the blueprint for the `fastapi-app` image. It should be a multi-stage build for efficiency.
  - **Stage 1 (Builder):** Installs `requirements.txt` into a virtual environment.
  - **Stage 2 (Final):** Copies the `venv` and the `app/` source code from the builder. Runs the `uvicorn` server.
- **`docker-compose.yml`**: Defines the local development stack.
  - **`app` service:** Builds from the `Dockerfile`. Mounts the local `app` directory as a volume for hot-reloading. Depends on the `db` service.
  - **`db` service:** Pulls the official `postgres:latest` image. Sets environment variables for the database from the `.env` file.

### Kubernetes

- **`db-deployment.yaml`**: Deploys PostgreSQL.
  - **Note:** For production, this should be a `StatefulSet` with a `PersistentVolumeClaim` (PVC) to ensure data is not lost when the pod restarts. The current file is a simple `Deployment`, which is fine for testing but not production.
- **`db-service.yaml`**: Exposes the database _within_ the cluster (using a `ClusterIP`) so the `app` pods can find it.
- **`deployment.yaml`**: Deploys the `python-app-deployment`.
  - Manages the `ReplicaSet`.
  - Pulls the `python-app-deployment` image .
  - Configures environment variables by referencing secrets (like `GEMINI_API_KEY`) and `ConfigMap`s.
- **`app-service.yaml`**: Exposes the `python-app-deployment` to the outside world.
  - Typically uses a `LoadBalancer` (on cloud).

---

## 6. 🔑 Environment Variables

This is a central list of all required environment variables.

| Variable            | `docker-compose.yml` |    Kubernetes    | `local .env` | Description                             |
| :------------------ | :------------------: | :--------------: | :----------: | :-------------------------------------- |
| `POSTGRES_USER`     |       **Yes**        | **Yes** (Secret) |   **Yes**    | Username for PostgreSQL.                |
| `POSTGRES_PASSWORD` |       **Yes**        | **Yes** (Secret) |   **Yes**    | Password for PostgreSQL.                |
| `POSTGRES_DB`       |       **Yes**        | **Yes** (Secret) |   **Yes**    | Database name.                          |
| `POSTGRES_HOST`     |     **Yes** (db)     |     **Yes**      |      No      | Hostname of the DB.                     |
| `DATABASE_URL`      |          No          |        No        |   **Yes**    | Full connection string (for local dev). |
| `GEMINI_API_KEY`    |       **Yes**        | **Yes** (Secret) |   **Yes**    | Your Google Gemini API Key.             |
