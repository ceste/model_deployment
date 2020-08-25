import config

import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from pipeline import Pipeline



class Prediction(Pipeline):

	def __init__(self, dataframe, random_state=42):

		Pipeline.__init__(self, dataframe, random_state)

		self.random_state = random_state

		self.dataframe = dataframe

		self.prediction = None	

	def split_dataframe(self):

		feature_names = [col for col in self.dataframe.columns if col!=self.target_column]	

		data = self.dataframe.copy()

		X = data[feature_names]
		y = data[self.target_column]
		

		return X, y

	

	def decode_prediction(self, data):

		return super(Prediction, self).decode_target_feature(data)



	def predict(self):

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

		return self.prediction, prediction_labels

