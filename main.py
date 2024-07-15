import cv2

import model_integration, process_video, process_frame
from logging_utils import info_logger, error_logger



Local_WebCam_Stream = 'http://192.168.1.19:8000/'


def main():

    info_logger.info(" -- Person detection start -- ")

    model = model_integration.initialize_model()

    vidcap = process_video.capture_video(Local_WebCam_Stream)

    try:    
        while True:
                
            ret, frame = vidcap.read()

            if not ret:
                error_logger.error('Error : Failed to read frame -- break')
                break

            detected_objects = process_frame.object_detection(model, frame)

            for obj in detected_objects.itertuples():
                output_frame = process_frame.predictions_visualization(obj, frame)
                  
            cv2.imshow('Frame', output_frame)


            if cv2.waitKey(1) & 0xFF==ord('q'):
                info_logger.info(' -- Closing camera -- ')
                cv2.destroyAllWindows()
                break
                
    finally:
         vidcap.release()


if __name__ == "__main__":
    main()
