''' File including debugging and logging utilities to centralize log management and provide extra tools for debuging if needed'''

import logging

def setup_logger(name, log_file=None, level=logging.INFO):

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # file hadnler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)
    else: 
    # console handler 
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(console_handler)

    return logger


info_logger = setup_logger('info_logger')
debug_logger = setup_logger('debug_logger', level=logging.DEBUG)
error_logger = setup_logger('error_logger', level=logging.ERROR)

camera_logger = setup_logger('camera_logger', log_file='camera.log')
video_output_logger = setup_logger('video_output_logger', log_file='video_output.log')
