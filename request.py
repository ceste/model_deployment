import requests
import json
import config, utils
import argparse
import pandas as pd
from pathlib import Path
from log import Log

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

	# # local url
	url = config.URL

	# heroku url
	# url = config.HEROKU_URL

	# data dummy
	dataset_path = Path(config.PATH_TO_DATASET+'/'+config.DATASET)	
	json_data = pd.read_csv(dataset_path)[0:1].to_json(orient='records')[1:-1].replace('},{', '} {')

	json_data = json.loads(json_data)

	print('Predict:',json_data)		

	msg = 'Predict:',json_data
	log.print(msg)

	json_data['user_id'] = user_id
	
	data = json.dumps(json_data)

	print(url)
	r_survey = requests.post(url, data)
	print(r_survey)

	send_request = requests.post(url, data)
	print(send_request)

	if send_request.status_code == 200:
		print(send_request.json())


