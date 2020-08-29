import config, utils

from log import Log

import pandas as pd
import numpy as np
import pickle

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


class Pipeline:
	
	def __init__(self, user_id, path_to_dataset, random_state=42,test_size=0.2):

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

		self.test_size = test_size		
		msg = 'test_size: ',self.test_size
		self.log.print(msg)

		self.version = config.VERSION
		msg = 'version: ',self.version
		self.log.print(msg)

		self.path_to_data_folder = config.PATH_TO_DATASET		
		msg = 'path_to_data_folder: ',self.path_to_data_folder
		self.log.print(msg)		

		self.dataframe = pd.read_csv(self.path_to_dataset)

		self.acceptable_columns = config.ACCEPTABLE_COLUMNS
		msg = 'acceptable_columns: ',self.acceptable_columns
		self.log.print(msg)		

		self.unnecessary_columns = config.UNNECESSARY_COLUMNS
		msg = 'unnecessary_columns: ',self.unnecessary_columns
		self.log.print(msg)		

		self.categorical_columns = config.CATEGORICAL_COLUMNS
		msg = 'categorical_columns: ',self.categorical_columns
		self.log.print(msg)		

		self.numerical_columns = config.NUMERICAL_COLUMNS
		msg = 'numerical_columns: ',self.numerical_columns
		self.log.print(msg)		

		self.target_column = config.TARGET_COLUMN
		msg = 'target_column: ',self.target_column
		self.log.print(msg)		

		self.drop_these_features_first = config.DROP_THIS
		msg = 'drop_these_features_first: ',self.drop_these_features_first
		self.log.print(msg)		

		self.categorical_features = None
		file = Path(self.path_to_data_folder+'categorical_features_'+self.version+'.pkl')
		if file.is_file():
			self.categorical_features = self.load_pickle(file)    		

		self.numerical_features = None
		file = Path(self.path_to_data_folder+'numerical_features_'+self.version+'.pkl')
		if file.is_file():
			self.numerical_features = self.load_pickle(file)    		

		self.X_train = None
		self.X_valid = None
		self.y_train = None
		self.y_valid = None
		self.X_sm = None
		self.y_sm = None
		self.y_pred = None
		

		self.label_encoders = None
		file = Path(self.path_to_data_folder+'label_encoders_'+self.version+'.pkl')
		if file.is_file():
			self.label_encoders = self.load_pickle(file)    		

		self.one_hot_encoders = None
		file = Path(self.path_to_data_folder+'one_hot_encoders_'+self.version+'.pkl')
		if file.is_file():
			self.one_hot_encoders = self.load_pickle(file)

		self.dict = None
		file = Path(self.path_to_data_folder+'dict_'+self.version+'.pkl')
		if file.is_file():
			self.dict = self.load_pickle(file)

		self.model = None
		file = Path(self.path_to_data_folder+'model_rf_'+self.version+'.pkl')
		if file.is_file():
			self.model = self.load_pickle(file)


		# print(self.__class__.__name__)
		# print('Pipeline init')

		# log.print('self.__class__.__name__')

	def extract_features(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		try:
			self.dataframe = self.dataframe[self.acceptable_columns]
		except:
			print("There is a problem with the dataset")


		if self.dataframe.shape[1] != len(self.acceptable_columns):
			raise Exception("Number of columns is not valid. There is problem with the dataset.")

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)


	def replace_infinite_numbers(self, data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		# depend on your preference but this method works in this case
		data.replace([np.inf, -np.inf], np.nan, inplace=True)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return data		


	def handle_nan_values(self, x,y):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		# depend on your preference but this method works in this case
		X_cols = x.columns
		columns = list(X_cols) + list([self.target_column])		

		df = pd.concat([x, y], axis=1)	
		df.dropna(inplace=True)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return df[X_cols], df[self.target_column]		


	def drop_unnecessary_columns(self, data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		data.drop(self.unnecessary_columns, axis=1, inplace=True)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return data


	def validate_column_type(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		for col,type_ in zip(self.dataframe.columns,self.dataframe.dtypes):

			if col!=self.target_column:

				if (str(type_)=='object' and col in self.categorical_columns and col!=self.target_column) or (str(type_)!='object' and col in self.numerical_columns and col!=self.target_column):
					continue
				else:
					raise Exception("Column type for "+col+" is not valid.")
		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

	def features_engineering(self, data):		

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		data_copy = data.copy()

		data_copy['issue_d'] = pd.to_datetime(data_copy.loc[:,'issue_d'], infer_datetime_format=True)
		data_copy['issue_d_year'] = data_copy['issue_d'].apply(lambda x:str(x)[0:10].split('-')[0])
		data_copy['issue_d_year'] = data_copy['issue_d_year'].astype(int)

		data_copy['issue_d_month'] = data_copy['issue_d'].apply(lambda x:str(x)[0:10].split('-')[1])
		data_copy['issue_d_month'] = data_copy['issue_d_month'].astype(int)

		data_copy['issue_d_date'] = data_copy['issue_d'].apply(lambda x:str(x)[0:10].split('-')[2])
		data_copy['issue_d_date'] = data_copy['issue_d_date'].astype(int)

		data_copy['day_name'] = pd.Series(data_copy['issue_d']).dt.day_name()

		data_copy['loan_per_annual_inc'] = data_copy['loan_amount']/data_copy['annual_inc']
		data_copy['loan_per_annual_inc_cat'] = np.where(data_copy['loan_per_annual_inc']<=1,'<=1','>1')
		data_copy['installment_per_monthly_salary'] = data_copy['installment']/(data_copy['annual_inc']/12)
		data_copy['loan_per_annual_inc'] = data_copy['loan_amount']/data_copy['annual_inc']
		data_copy['installment_per_monthly_salary_cat'] = np.where(data_copy['installment_per_monthly_salary']>1,'>1','<=1')

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return data_copy


	def drop_this_first(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		self.dataframe.drop(self.drop_these_features_first, axis=1, inplace=True)	

	def split_dataframe(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		feature_names = [col for col in self.dataframe.columns if col!=self.target_column]	

		data = self.dataframe.copy()

		X = data[feature_names]
		y = data[self.target_column]
		
		self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

		pd.concat([self.X_train, self.y_train], axis=1).to_csv(self.path_to_data_folder+'train.csv')
		pd.concat([self.X_valid, self.y_valid], axis=1).to_csv(self.path_to_data_folder+'valid.csv')

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

	def get_ohe_column_names(self, dic,feature):
		return [feature+'_'+k for k,v in dic[feature].items() if v>0]
		

	def handle_categorical_columns(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		label_encoders = {}
		one_hot_encoders = {} 
		dic = {}

		for col in self.categorical_features:
			
			if col!=self.target_column:
				
				label_encoders[col] = LabelEncoder()        
				label_encoders[col].fit(self.X_train[col].values.reshape(-1,1))		        
				self.X_train[col] = label_encoders[col].transform(self.X_train[col])
				dic[col] = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))
											  
				one_hot_encoders[col] = OneHotEncoder(handle_unknown='ignore')
				one_hot_encoders[col].fit(self.X_train[col].values.reshape(-1,1))
				tmp = one_hot_encoders[col].transform(self.X_train[col].values.reshape(-1,1)).toarray()[:,1:]		        
				tmp_df = pd.DataFrame(tmp, columns=self.get_ohe_column_names(dic,col))
				
				self.X_train = pd.DataFrame(np.hstack([self.X_train,tmp_df]), columns=list(self.X_train.columns)+list(tmp_df.columns))
				
				del self.X_train[col]


		self.label_encoders = label_encoders
		self.one_hot_encoders = one_hot_encoders
		self.dict = dic

		self.save_as_pickle('label_encoders_'+self.version+'.pkl',self.label_encoders)
		self.save_as_pickle('one_hot_encoders_'+self.version+'.pkl',self.one_hot_encoders)
		self.save_as_pickle('dict_'+self.version+'.pkl',self.dict)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)
	

	def encode_categorical_data(self,data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		for col in self.categorical_features:

			if col!=self.target_column:

				# self.label_encoders[col].fit(data[col])
				data[col] = self.label_encoders[col].transform(data[col])				

				# self.one_hot_encoders[col].fit(data[col].values.reshape(-1,1))
				tmp = self.one_hot_encoders[col].transform(data[col].values.reshape(-1,1)).toarray()[:,1:]
				tmp_df = pd.DataFrame(tmp, columns=self.get_ohe_column_names(self.dict,col))

				data = pd.DataFrame(np.hstack([data,tmp_df]), columns=list(data.columns)+list(tmp_df.columns))
				del data[col]

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return data


	def load_pickle(self, filename):

		file = open(filename,'rb')
		object_file = pickle.load(file)
		file.close()
		return object_file

	def save_as_pickle(self,filename,data):
		
		filename = self.path_to_data_folder+filename
		with open(filename, 'wb') as fp:
			pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
		fp.close()

	def features_type_mapping(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		categorical_features = []
		numerical_features = []

		for col,type_ in zip(self.X_train.columns, self.X_train.dtypes):

			if col!=self.target_column:
				if str(type_)=='object' and col!=self.target_column:
					categorical_features.append(col)        
				else:
					numerical_features.append(col)

		self.categorical_features = categorical_features
		self.numerical_features = numerical_features

		self.save_as_pickle('categorical_features_'+self.version+'.pkl',self.categorical_features)
		self.save_as_pickle('numerical_features_'+self.version+'.pkl',self.numerical_features)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)


	def handle_target_feature(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		self.label_encoders[self.target_column] = LabelEncoder()        
		self.label_encoders[self.target_column].fit(self.y_train.values)
		self.y_train = self.label_encoders[self.target_column].transform(self.y_train.values)		
		self.save_as_pickle('label_encoders_'+self.version+'.pkl',self.label_encoders)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)


	def encode_target_feature(self,data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)
		
		output = self.label_encoders[self.target_column].transform(data.values)
		return output

	def decode_target_feature(self,data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)
		
		output = self.label_encoders[self.target_column].inverse_transform(data)
		return output

	def upsampling(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		smote = SMOTE()		
		self.X_sm, self.y_sm = smote.fit_sample(self.X_train, self.y_train)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

	def train_model(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		rf = RandomForestClassifier(random_state=42)
		rf.fit(self.X_sm, self.y_sm.reshape(-1,1))

		self.model = rf

		#save model
		self.save_as_pickle('model_rf_'+self.version+'.pkl',self.model)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)


	def predict(self, data):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)

		return self.model.predict(data)

	def evaluate_model(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		print(confusion_matrix(self.y_valid, self.y_pred))
		print("Accuracy:",accuracy_score(self.y_valid, self.y_pred))
		print('AUC:',roc_auc_score(self.y_valid,self.y_pred))
		print('Precision:',precision_score(self.y_valid,self.y_pred))
		print('Average Precision Score:',average_precision_score(self.y_valid, self.y_pred))
		print('Recall:',recall_score(self.y_valid,self.y_pred))
		print('F1 Score:',f1_score(self.y_valid,self.y_pred))

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)



	def train(self):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> enter'
		self.log.print(msg)

		self.extract_features()

		self.validate_column_type()

		self.drop_this_first()

		self.split_dataframe()

		self.X_train = self.features_engineering(self.X_train)
		self.X_valid = self.features_engineering(self.X_valid)

		self.X_train = self.replace_infinite_numbers(self.X_train)
		self.X_valid = self.replace_infinite_numbers(self.X_valid)
				
		self.X_train, self.y_train = self.handle_nan_values(self.X_train,self.y_train)
		self.X_valid, self.y_valid = self.handle_nan_values(self.X_valid,self.y_valid)
		
		self.X_train = self.drop_unnecessary_columns(self.X_train)
		self.X_valid = self.drop_unnecessary_columns(self.X_valid)
	
		self.features_type_mapping()

		self.handle_categorical_columns()		
		self.X_valid = self.encode_categorical_data(self.X_valid)

		self.handle_target_feature()
		self.y_valid = self.encode_target_feature(self.y_valid)

		self.upsampling()

		self.train_model()

		self.y_pred = self.predict(self.X_valid)

		self.evaluate_model()

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+' -> exit'
		self.log.print(msg)