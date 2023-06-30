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
from src.utils import save_object

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

            #Reading the train and the test files using pandas
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            #Load the preprocessing transformer object
            preprocessing_obj = self.get_data_transformer_object()


            #Defining the X and the y dataframes for training
            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = train_df[target_column_name]


            #Preprocessing of the X feature
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Final train and test array after concatenation with target variable
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            #Save the object
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        



            



