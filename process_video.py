''' This file focus on handling input values from camera device '''

import cv2
from typing import Optional

from logging_utils import info_logger, camera_logger, camera_log_pattern


def capture_video(url: str) -> Optional[cv2.VideoCapture]:
    '''
    Open a video capture stream from the specified url. 

    :param url: url as a local network address pointing to a video stream source
    :return: If successfully opened, a VideoCapture object from which read frames, otherwise None.
    
    '''
    
    vidcap = cv2.VideoCapture(url)

    if vidcap.isOpened():
        info_logger.info(' -- Camera is open -- ')
        camera_log_pattern(camera_logger, vidcap)
        return vidcap
    else:
        info_logger.error('Failed to open camera')
        raise
