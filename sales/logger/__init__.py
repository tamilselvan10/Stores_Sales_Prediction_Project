import logging
import os,sys
from datetime import datetime

LOG_DIR='logs'
CURRENT_TIME_STAMP=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
LOG_FILE_NAME=f'log_{CURRENT_TIME_STAMP}.log'

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode='w',
format='[%(asctime)s]:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s',
level=logging.INFO
)