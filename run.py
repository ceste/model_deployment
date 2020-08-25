import pandas as pd
import numpy as np

import config
from pipeline import Pipeline
from prediction import Prediction

from pathlib import Path

if __name__ == '__main__':

	random_state = 42    
	
	dataset_path = Path(config.PATH_TO_DATASET+'/'+config.DATASET)	
	if dataset_path.exists():
		

		data = pd.read_csv(dataset_path)


		pipeline = Pipeline(dataframe = data, random_state=random_state, test_size=0.2)
		pipeline.train()

		prediction = Prediction(dataframe = data[0:100], random_state=random_state)
		predictions, labels = prediction.predict()

		print('predictions:',predictions)
		print('prediction labels:',labels)	
		print('ground truth:',data[config.TARGET_COLUMN])



	else:
		print('file is not exist')

