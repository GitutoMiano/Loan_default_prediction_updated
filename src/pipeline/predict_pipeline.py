import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", 'models', "lgbm_model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = None
        self.preprocessor = None
        self.data_transformation = DataTransformation()

    def _load_resources(self):
        try:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not os.path.isfile(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")
            
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
        
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            if self.model is None or self.preprocessor is None:
                self._load_resources()

            # Rename columns to match those used during training
            features = self.data_transformation.col_rename(features)

            # Ensure the columns in features match the preprocessor's expected columns
            expected_columns = self.preprocessor.feature_names_in_
            missing_columns = set(expected_columns) - set(features.columns)
            if missing_columns:
                raise KeyError(f"Missing columns in input features: {missing_columns}")

            # Reorder columns to match preprocessor expectation
            features = features[expected_columns]

            # Transform features and make predictions
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(
        self,
        loan_limit: str,
        gender: str,
        approv_in_adv: str,
        loan_type: str,
        loan_purpose: str,
        credit_worthiness: str,
        open_credit: str,
        business_or_commercial: str,
        loan_amount: float,
        rate_of_interest: float,
        interest_rate_spread: float,
        upfront_charges: float,
        term: float,
        neg_ammortization: str,
        property_value: float,
        construction_type: str,
        occupancy_type: str,
        secured_by: str,
        security_type: str,
        purpose: str,
        occupancy: str,
        credit_score: int,
        co_applicant_credit_type: str,
        age: int,
        submission_of_application: str,
        ltv: float,
        region: str,
        dtir1: float,
        income: float,
    ):
        # Initialize the attributes with input values
        self.loan_limit = loan_limit
        self.gender = gender
        self.approv_in_adv = approv_in_adv
        self.loan_type = loan_type
        self.loan_purpose = loan_purpose
        self.credit_worthiness = credit_worthiness
        self.open_credit = open_credit
        self.business_or_commercial = business_or_commercial
        self.loan_amount = loan_amount
        self.rate_of_interest = rate_of_interest
        self.interest_rate_spread = interest_rate_spread
        self.upfront_charges = upfront_charges
        self.term = term
        self.neg_ammortization = neg_ammortization
        self.property_value = property_value
        self.construction_type = construction_type
        self.occupancy_type = occupancy_type
        self.secured_by = secured_by
        self.security_type = security_type
        self.purpose = purpose
        self.occupancy = occupancy
        self.credit_score = credit_score
        self.co_applicant_credit_type = co_applicant_credit_type
        self.age = age
        self.submission_of_application = submission_of_application
        self.ltv = ltv
        self.region = region
        self.dtir1 = dtir1
        self.income = income

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with feature names and values
            custom_data_input_dict = {
                "loan_limit": [self.loan_limit],
                "gender": [self.gender],
                "approv_in_adv": [self.approv_in_adv],
                "loan_type": [self.loan_type],
                "loan_purpose": [self.loan_purpose],
                "credit_worthiness": [self.credit_worthiness],
                "open_credit": [self.open_credit],
                "business_or_commercial": [self.business_or_commercial],
                "loan_amount": [self.loan_amount],
                "rate_of_interest": [self.rate_of_interest],
                "interest_rate_spread": [self.interest_rate_spread],
                "upfront_charges": [self.upfront_charges],
                "term": [self.term],
                "neg_ammortization": [self.neg_ammortization],
                "property_value": [self.property_value],
                "construction_type": [self.construction_type],
                "occupancy_type": [self.occupancy_type],
                "secured_by": [self.secured_by],
                "security_type": [self.security_type],
                "purpose": [self.purpose],
                "occupancy": [self.occupancy],
                "credit_score": [self.credit_score],
                "co_applicant_credit_type": [self.co_applicant_credit_type],
                "age": [self.age],
                "submission_of_application": [self.submission_of_application],
                "ltv": [self.ltv],
                "region": [self.region],
                "dtir1": [self.dtir1],
                "income": [self.income],
            }

            # Convert the dictionary into a pandas DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            return df
        
        except Exception as e:
            raise CustomException(e, sys)
