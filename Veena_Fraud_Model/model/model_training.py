from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
# import lightgbm as lgb


class ModelTrainer:
    def __init__(self):
        """Initialize trainer without manual scaling."""
        self.models = {}
        self.results = {}

    def add_models(self):
        """Define a dictionary of models."""
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42),
            # "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
            # "LightGBM": lgb.LGBMClassifier()
        }

    def fit(self, X_train, y_train, X_test, y_test, preprocessor):
        self.results = {}
        self.trained_models = {}
        for name, model in self.models.items():
            # Pipeline: Preprocessor -> SMOTE -> Model
            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            self.results[name] = {
                "pipeline": pipeline,
                "accuracy": report_dict["accuracy"],
                "precision": report_dict["weighted avg"]["precision"],
                "recall": report_dict["weighted avg"]["recall"],
                "f1": report_dict["weighted avg"]["f1-score"],
                "report": classification_report(y_test, y_pred)
            }
            self.trained_models[name] = pipeline
            print(f"\nClassification Report for {name}:")
            print(self.results[name]["report"])
        return self.results

    def get_best_model(self):
        best_name = max(self.results, key=lambda x: self.results[x]["f1"])
        best_model = self.trained_models[best_name]
        best_metrics = self.results[best_name]
        return best_name, best_model, best_metrics
