import re
import pandas as pd
from preprocess import ClaimPreprocessor
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

def main():
    # Load and preprocess
    preprocessor = ClaimPreprocessor("C:/Users/Veena SP/OneDrive/Desktop/Allianz/Fruad_api/data/carclaims.csv")
    raw_df = preprocessor.load_data()
    print("Raw shape:", raw_df.shape)
    processed_df = preprocessor.preprocess()
    print("Processed shape:", processed_df.shape)

    # Features and target
    X = processed_df.drop("FraudFound", axis=1)
    y = processed_df["FraudFound"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor pipeline
    categorical_features = ["Month", "DayOfWeek", "Make", "AccidentArea", "Sex", "MaritalStatus", "PolicyType","VehicleCategory", "VehiclePrice"]
    numeric_features = ["Age", "NumberOfCars"]
    preprocessor_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ])

    # Train models with SMOTE
    trainer = ModelTrainer()
    trainer.add_models()
    results = trainer.fit(X_train, y_train, X_test, y_test, preprocessor_pipeline)

    # Print summary
    print("\nModel Performance Summary:")
    for name, metrics in results.items():
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # Best model selection
    best_name, best_pipeline, best_metrics = trainer.get_best_model()
    print(f"\nBest Model: {best_name} with F1-score={best_metrics['f1']:.4f}")

    # Save full pipeline
    joblib.dump(best_pipeline, "C:/Users/Veena SP/OneDrive/Desktop/Allianz/Fruad_api/final_pipeline.pkl")
    print("Full pipeline saved as final_pipeline.pkl")


if __name__ == "__main__":
    main()
