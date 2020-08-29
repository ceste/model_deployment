import config, utils
from log import Log
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from pipeline import Pipeline



class Prediction(Pipeline):
	
	def __init__(self, user_id, path_to_dataset, random_state=42):

		Pipeline.__init__(self, user_id, path_to_dataset, random_state)		

		self.log = Log()
		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)


		self.user_id = user_id
		msg = 'user_id: ',self.user_id
		self.log.print(msg)

		self.path_to_dataset = path_to_dataset
		msg = 'path_to_dataset: ',self.path_to_dataset
		self.log.print(msg)

		self.random_state = random_state
		msg = 'random_state: ',self.random_state
		self.log.print(msg)		

		self.dataframe = pd.read_csv(self.path_to_dataset)

		self.prediction = None	

	def split_dataframe(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		feature_names = [col for col in self.dataframe.columns if col!=self.target_column]	

		data = self.dataframe.copy()

		X = data[feature_names]
		y = data[self.target_column]
		
		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)


		return X, y

	

	def decode_prediction(self, data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)		

		return super(Prediction, self).decode_target_feature(data)



	def predict(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		super(Prediction, self).extract_features()

		super(Prediction, self).validate_column_type()

		super(Prediction, self).drop_this_first()

		self.X, self.y = self.split_dataframe()		

		self.X = super(Prediction, self).features_engineering(self.X)		

		self.X = super(Prediction, self).replace_infinite_numbers(self.X)		
				
		self.X, self.y = super(Prediction, self).handle_nan_values(self.X,self.y)		
		
		self.X = super(Prediction, self).drop_unnecessary_columns(self.X)		
				
		self.X = super(Prediction, self).encode_categorical_data(self.X)
		
		self.y = super(Prediction, self).encode_target_feature(self.y)			

		self.prediction = super(Prediction, self).predict(self.X)

		prediction_labels = self.decode_prediction(self.prediction)		

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return self.prediction, prediction_labels

		