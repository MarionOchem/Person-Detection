''' This file focus on processing the informations for the device and making it ready to output / use it '''

import cv2
from typing import Optional
import torch
import numpy as np
import pandas as pd

from logging_utils import debug_logger


def object_detection(model: torch.nn.Module, image: np.ndarray) -> Optional[pd.DataFrame]:
    '''
    Perform object detection on an input image using a given PyTorch model.

    :param model: YOLOv5 configured for object detection.
    :param image: Input image process as a NumPy array by cv2 : a 2-dimension matrix of RGB pixel value.
    :return: DataFrame containing detected objects' bounding box coordinates,
             confidence scores, and class labels. Returns None if no objects are detected

    Note : 
    1. Machine state : uncomment debug_logger.debug to log detailed informations about frame input and detected objects for each frame in terminal.
    2. Model contains input handling mechanisms built-in converting image to tensors [batch_size, channels, height, width] ->  torch.as_tensor() ?
    '''
    # debug_logger.debug(f'Model input : {image}')
    preds = model(image)
    detectedObjects = preds.pandas().xyxy[0]
    # debug_logger.debug(detected_objects)

    return detectedObjects


def predictions_visualization(data: pd.Series, image: np.ndarray) -> np.ndarray:
    '''
    Create predictions visualization by drawing boudning boxes and labels on frame. 

    :param data: Pandas Series containing bouding box coordinates and confidence score.
    :param image: Input image as a NumPy array on which to draw visualizations.
    :return: Image with bounding boxes and labels drawn. 

    '''
   
    # Draw bounding box
    cv2.rectangle(image, (int(data.xmin), int(data.ymin)), (int(data.xmax), int(data.ymax)), (0, 0, 255), 2)  

    ''' 
    cv.rectangle( img, pt1, pt2, color[, thickness[, lineType[, shift]]] ) ->	img

    '''
    
    # Draw label and confidence
    cv2.putText(image, f"{data.name} : {data.confidence:.2f}", (int(data.xmin), int(data.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

    '''
    cv.putText(	img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	) ->	img

    '''

    return image
