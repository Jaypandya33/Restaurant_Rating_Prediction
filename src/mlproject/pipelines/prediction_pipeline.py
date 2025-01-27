import os
import sys
import pandas as pd
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
from src.mlproject.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    

class CustomData:
    def __init__(self,
                 online_order:str,
                 book_table:str,
                 votes:float,
                 location:str,
                 rest_type:str,
                 cuisines:str,
                 cost_for_two_people:float,
                 type:str
                 ):
        
        self.online_order=online_order
        self.book_table=book_table
        self.votes=votes
        self.location=location
        self.rest_type=rest_type
        self.cuisines=cuisines
        self.cost_for_two_people=cost_for_two_people
        self.type=type
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'online_order':[self.online_order], 
                    'book_table':[self.book_table], 
                    'votes':[self.votes], 
                    'location':[self.location], 
                    'rest_type':[self.rest_type], 
                    'cuisines':[self.cuisines],
                    'cost_for_two_people':[self.cost_for_two_people],
                    'type':[self.type]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)

