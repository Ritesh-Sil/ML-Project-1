import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
#Column transformer is used for pipeline purpose

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        ''' This function is responsible for data transformation'''
        try:
            numerical_column = [
                'reading_score',
                'writing_score'

            ]

            categorical_column = [

                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'

            ]

            num_pipeline =Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()) 
                ]
            )


            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"Categorical columns:{categorical_column}")
            logging.info(f"Categorical columns: {categorical_column}")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )


            return preprocessor
        


        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self, train_path, test_path):

        try:
            pass

        except Exception as e:
            pass
            



