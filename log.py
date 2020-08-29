import logging 
import config 

logger = logging.getLogger(__name__)
logging.basicConfig(filename=config.PATH_TO_DATASET+"loging_"+config.VERSION+".log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")


class Log():
    def __init__(self):        

    	pass

    def print(self, message):
        logging.info("{} ".format(message))
    
