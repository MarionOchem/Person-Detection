''' File including debugging and logging utilities to centralize log management and provide extra tools for debuging if needed'''

import logging
import cv2


def setup_logger(name, log_file=None, level=logging.INFO):

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    logger.handlers.clear()
    
    # file handler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(f'%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)
    else: 
    # console handler 
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(f'%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(console_handler)

    # Prevent propagation to parent loggers
    logger.propagate = False

    return logger



def model_log_pattern(logger, model):

    # Model Summary 
    logger.info(f'MODEL SUMMARY :\n{model}')

    logger.info('')

    # Model Parameters
    logger.info('MODEL PARAMETERS :')
    for name, param in model.named_parameters():
        logger.info(f'Parameter : {name}, Size : {param.size}')

    logger.info('')

    # Model Configuration
    logger.info(f'MODEL CONFIGURATION :')
    logger.info(f'Model classes : {model.classes}')
    logger.info(f'Confidence threshold : {model.conf}')

    # Model device 
    device = next(model.parameters()).device
    logger.info(f'Model device : {device}')


def camera_log_pattern(logger, video_vapture):

    # Video capture size
    frame_width = video_vapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_vapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f'Frame width: {frame_width}')
    logger.info(f'Frame height: {frame_height}')

    # Video capture performance
    fps = video_vapture.get(cv2.CAP_PROP_FPS)
    logger.info(f'FPS: {fps}')


    # Video capture data format
    codec = video_vapture.get(cv2.CAP_PROP_FOURCC)
    codec_str = ''.join([chr((int(codec) >> 8 * i) & 0xFF) for i in range(4)])  # Convert codec to string
    logger.info(f'Codec: {codec_str}')



# Terminal log
info_logger = setup_logger('info_logger')
debug_logger = setup_logger('debug_logger', level=logging.DEBUG)
error_logger = setup_logger('error_logger', level=logging.ERROR)

# File log
model_logger = setup_logger('model_logger', log_file='model.log', level=logging.DEBUG)
camera_logger = setup_logger('camera_logger', log_file='camera.log', level=logging.DEBUG)
# video_output_logger = setup_logger('video_output_logger', log_file='video_output.log')



# TODO: create a function to setup logger detail specific for the model logger output file