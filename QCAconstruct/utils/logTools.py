# -*- coding: utf-8 -*-
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',  
)

# Create a logger for fileID
logger_fileID = logging.getLogger('logger_fileID') 
logger_fileID_Hanlder = logging.FileHandler("./logs/generation.log")
logger_fileID_Hanlder.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
logger_fileID.addHandler(logger_fileID_Hanlder)
logger_fileID.setLevel(logging.INFO)

# Create a logger for process
logger_process = logging.getLogger('logger_process')  
logger_process_Hanlder = logging.FileHandler("./logs/process.log")
logger_process_Hanlder.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
logger_process.addHandler(logger_process_Hanlder)
logger_process.setLevel(logging.INFO)