import requests
import json
import config

import pandas as pd
from pathlib import Path

if __name__ == '__main__':

	# local url
	url = config.URL

	# data dummy
	dataset_path = Path(config.PATH_TO_DATASET+'/'+config.DATASET)	
	json_data = pd.read_csv(dataset_path)[0:1].to_json(orient='records')[1:-1].replace('},{', '} {')
	json_data = json.loads(json_data)
	print('Predict:',json_data)	

	data = json.dumps(json_data)

	r_survey = requests.post(url, data)
	print(r_survey)

	send_request = requests.post(url, data)
	print(send_request)

	print(send_request.json())


