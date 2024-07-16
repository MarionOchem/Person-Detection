''' This file configure the YOLOv5 model from PyTorch Hub API for this inference usage '''

import torch
import intel_extension_for_pytorch as ipex 
from typing import Optional

from logging_utils import info_logger, error_logger, model_logger, model_log_pattern


def initialize_model(retry_attemps: int = 3) -> Optional[torch.nn.Module]:
    '''
    Load a YOLOv5 model from the "ultralytics/yolov5" repository using Torch Hub.

    :param retry_attempts: Number of attempts to retry model initialization in case of error. Default is 3.

    '''

    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.eval()        # Switches to evaludation mode / inference
        model.classes = [0]         # Configure model to detect only class "Person" 
        model.conf = 0.6        # Configure model to detect only predicions >= 0.6 confidence score
        model = ipex.optimize(model, dtype=torch.float32)       # operator pattern fusion to improve the efficiency of model execution. 

        model_log_pattern(model_logger, model)

        info_logger.info(' -- Model succesfully integrated -- ')
        return model
    
    except Exception as e:

        if retry_attemps > 0:
            error_logger.error(f'Error initializing model : {e} \n Retrying initialization. Attempts left : {retry_attemps - 1}')
            return initialize_model(retry_attemps - 1)
        else:
            error_logger.error('Failed to initialize model after multiple attempts. Stopping program.')
            raise RuntimeError