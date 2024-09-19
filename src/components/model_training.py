import os
import sys
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.utils import save_object

# Define the path to the artifacts directory
artifacts_dir = "artifacts"

# Create the directory if it does not exist
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

class ModelTrainerConfig:
    def __init__(self):
        self.trained_models_dir = os.path.join(artifacts_dir, "models")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def save_feature_importances(self, model, feature_names, filename):
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            feature_importance_df.to_csv(filename, index=False)
        else:
            print(f"Model {type(model).__name__} does not support feature importances")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=7215),
                'Random Forest': RandomForestClassifier(random_state=7215),
                'XGBoost': XGBClassifier(random_state=7215),
                'AdaBoost': AdaBoostClassifier(random_state=7215),
                'LGBM': LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                    importance_type='split', learning_rate=0.1, max_depth=-1,
                                    min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                    n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                                    random_state=5893, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
                                    subsample_for_bin=200000, subsample_freq=0)}

            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

            for model_name, model in models.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                print(f"\nClassification Report for {model_name}:\n")
                print(classification_report(y_test, y_pred))

                print(f"{model_name} - Accuracy: {accuracy}")
                print(f"{model_name} - Precision: {precision}")
                print(f"{model_name} - Recall: {recall}")
                print(f"{model_name} - F1 Score: {f1}")

                # Save the trained model
                model_path = os.path.join(self.model_trainer_config.trained_models_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                save_object(file_path=model_path, obj=model)

                # Save feature importances if available
                feature_importance_path = os.path.join(self.model_trainer_config.trained_models_dir, f"{model_name.lower().replace(' ', '_')}_feature_importances.csv")
                self.save_feature_importances(model, feature_names, feature_importance_path)

        except Exception as e:
            raise CustomException(e, sys)
