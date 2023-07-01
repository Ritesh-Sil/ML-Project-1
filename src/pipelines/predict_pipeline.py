import sys
import os

import pandas as pd
import numpy as np

from src.exception import Exception
from src.utils import load_object



class PrectPipeline:
    def __init__(self):
        pass


# Mapping the inputs in the html with the backend values
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender,
        self.race_ethnicity = race_ethnicity,
        self.    






