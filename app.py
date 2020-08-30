import pandas as pd
import json
from pandas.io.json import json_normalize

from flask import Flask, jsonify, request

import pickle
import config, utils

from prediction import Prediction
from log import Log

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():

	log = Log()
	msg = __name__+'.'+utils.get_function_caller()+' -> enter'
	log.print(msg)
	
	# get data
	json_data = request.get_json(force=True)

	input_df = pd.json_normalize(json_data)

	# save json_data and input_df for debugging purpose, save using unique name
	json_data_unique_filename = config.PATH_TO_DATASET+utils.get_unique_filename('json_data.json')
	input_df_unique_filename = config.PATH_TO_DATASET+utils.get_unique_filename('input_df.csv')		

	with open(json_data_unique_filename, 'w') as outfile:
		json.dump(json_data, outfile)

	input_df.to_csv(input_df_unique_filename, index=False)

	user_id = 'cst'
	random_state = 42

	prediction = Prediction(user_id, input_df_unique_filename, random_state)
	predictions, labels = prediction.predict()

	result = {'prediction': int(predictions[0]), 'label': str(labels[0])}
	# output = {'result': result}
	output = result

	msg = __name__+'.'+utils.get_function_caller()+' -> exit'
	log.print(msg)	
	
	return jsonify(results=output)

if __name__ == '__main__':
	app.run(port = 5000, debug=True)

