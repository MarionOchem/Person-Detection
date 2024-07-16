import cv2

import model_initialization, process_video, process_frame
from logging_utils import info_logger, error_logger



LocalWebCamStream = 'http://192.168.1.19:8000/'


def main():
    '''
    Detecting persons using a model on a live local video stream and generating a video with visualized predictions.

    '''

    info_logger.info(" -- Person detection start -- ")

    # Initialize the model for person detection
    model = model_initialization.initialize_model()

    # Capture video stream from the local webcam
    vidcap = process_video.capture_video(LocalWebCamStream)

    try: 
        # Main loop to read frames from video stream and perform object detection visualization  
        while True:
            
            # Read a frame from the video stream
            ret, frame = vidcap.read() 
            '''
            .read() output a tuple
            ret = boolean indicating if a frame was successfully read
            frame = NumPy array representing the image frame
            '''

            if not ret:
                error_logger.error('Error : Failed to read frame -- break')
                break
            
            # Perform object detection 
            detectedObjects = process_frame.object_detection(model, frame)

            # Visualize predictions on the frame
            for obj in detectedObjects.itertuples():
                outputFrame = process_frame.predictions_visualization(obj, frame)
            
            # Display the frame with predictions
            cv2.imshow('Frame', outputFrame)

            # Check for user input to quit the program
            if cv2.waitKey(1) & 0xFF==ord('q'):
                info_logger.info(' -- Closing camera -- ')
                cv2.destroyAllWindows()
                break
                
    finally:
         vidcap.release()


if __name__ == "__main__":
    main()
