''' This file focus on handling input values from camera device '''

import cv2
from typing import Optional

from logging_utils import info_logger, camera_logger, camera_log_pattern


def capture_video(url: str) -> Optional[cv2.VideoCapture]:

    vidcap = cv2.VideoCapture(url)

    if vidcap.isOpened():
        info_logger.info(' -- Camera is open -- ')
        camera_log_pattern(camera_logger, vidcap)
        return vidcap
    else:
        info_logger.error('Failed to open camera')
        raise
