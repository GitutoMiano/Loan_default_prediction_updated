from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def col_rename(self, df):
        """
        Renames columns by replacing spaces and hyphens with underscores and converting to lowercase.
        """
        for column in df.columns:
            df.rename(columns={column: column.replace(" ", "_").replace("-", "_").lower()}, inplace=True)
        return df    

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['loan_amount', 'rate_of_interest', 'interest_rate_spread',
                                 'upfront_charges', 'term', 'property_value', 'income',
                                 'credit_score', 'ltv', 'dtir1']
            categorical_columns = ['loan_limit', 'gender', 'approv_in_adv', 'loan_type',
                                   'loan_purpose', 'credit_worthiness', 'open_credit',
                                    'business_or_commercial', 'neg_ammortization', 'interest_only',
                                    'lump_sum_payment','construction_type', 'occupancy_type',
                                    'secured_by', 'total_units', 'credit_type',
                                    'co_applicant_credit_type', 'age', 'submission_of_application',
                                    'region', 'security_type']

            num_pipeline = Pipeline(
                steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Drop unnecessary columns
            columns_to_drop = ['ID', 'year']  
            train_df = train_df.drop(columns=columns_to_drop, axis=1)
            test_df = test_df.drop(columns=columns_to_drop, axis=1)

            # Apply column renaming
            train_df = self.col_rename(train_df)
            test_df = self.col_rename(test_df)

            # Print columns for debugging
            logging.info(f"Train DataFrame columns: {train_df.columns}")
            logging.info(f"Test DataFrame columns: {test_df.columns}")

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column name and numerical columns
            target_column_name = "status"
            numerical_columns = ['loan_amount', 'rate_of_interest', 'interest_rate_spread',
                                'upfront_charges', 'term', 'property_value', 'income',
                                'credit_score', 'ltv', 'dtir1']

            # Check if target column and numerical columns are present in the DataFrame
            if target_column_name not in train_df.columns:
                raise KeyError(f"Target column '{target_column_name}' not found in training data.")
            if not set(numerical_columns).issubset(train_df.columns):
                missing_columns = set(numerical_columns) - set(train_df.columns)
                raise KeyError(f"Numerical columns missing from training data: {missing_columns}")

            # Prepare features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
