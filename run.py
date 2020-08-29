import pandas as pd
import numpy as np

import argparse

import json
from pandas.io.json import json_normalize

import config
import utils
from pipeline import Pipeline
from prediction import Prediction
from log import Log
from pathlib import Path

if __name__ == '__main__':


	log = Log()
	msg = __name__+'.'+utils.get_function_caller()+' -> enter'
	log.print(msg)

	parser = argparse.ArgumentParser()

	parser.add_argument("--user", "-u", help="set user id for tracking", required=True)

	args = parser.parse_args()

	msg = 'args: ',args
	log.print(msg)

	user_id = args.user


	random_state = 42    
	
	dataset_path = Path(config.PATH_TO_DATASET+'/'+config.DATASET)	
	if dataset_path.exists():		

		pipeline = Pipeline(user_id=user_id, path_to_dataset=dataset_path, random_state=random_state, test_size=0.2)
		pipeline.train()
	
		# data dummy for prediction demo
		# should be in json format as it will be fed into API, will be managed by controlled later on
		json_data = pd.read_csv(dataset_path)[0:1].to_json(orient='records')[1:-1].replace('},{', '} {')
		json_data = json.loads(json_data)
		print('Predict:',json_data)

		# controller will convert json data to dataframe		
		input_df = pd.json_normalize(json_data)

		# save json_data and input_df for debugging purpose, save using unique name
		json_data_unique_filename = config.PATH_TO_DATASET+utils.get_unique_filename('json_data.json')
		input_df_unique_filename = config.PATH_TO_DATASET+utils.get_unique_filename('input_df.csv')		
		
		with open(json_data_unique_filename, 'w') as outfile:
			json.dump(json_data, outfile)

		input_df.to_csv(input_df_unique_filename, index=False)

		
		prediction = Prediction(user_id, input_df_unique_filename, random_state)
		predictions, labels = prediction.predict()

		print('predictions:',predictions)
		print('prediction labels:',labels)	
		print('ground truth:',json_data[config.TARGET_COLUMN])

		

	else:
		print('file is not exist')


	msg = __name__+'.'+utils.get_function_caller()+' -> exit'
	log.print(msg)