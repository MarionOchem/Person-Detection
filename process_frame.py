''' This file focus on processing the informations for the device and making it ready to output / use it '''

import cv2
from typing import Optional
import torch
import numpy as np
import pandas as pd
import logging

from logging_utils import debug_logger


def object_detection(model: torch.nn.Module, image: np.ndarray) -> Optional[pd.DataFrame]:
    preds = model(image)
    detected_objects = preds.pandas().xyxy[0]
    # debug_logger.info(detected_objects)

    return detected_objects


def predictions_visualization(data: pd.Series, image: np.ndarray) -> np.ndarray:
   
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
