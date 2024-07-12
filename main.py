import cv2

import model_integration, process_video, process_frame



def main():

    print(" -- Person detection start -- ")

    model = model_integration.initialize_model()

    vidcap = process_video.capture_video("http://192.168.1.19:8000/")

    try:    
        while True:
                
            ret, frame = vidcap.read()

            if not ret:
                print("Error : Failed to read frame")
                break

            detected_objects = process_frame.object_detection(model, frame)

            for obj in detected_objects.itertuples():
                process_frame.predictions_visualization(obj, frame)
                  
            cv2.imshow("Frame", frame)


            if cv2.waitKey(1) & 0xFF==ord('q'):
                print("Closing camera")
                cv2.destroyAllWindows()
                break
                
    finally:
         vidcap.release()


if __name__ == "__main__":
    main()



    
    # if vidcap.isOpened():
    #     print("Cam open")

    #     while True:
            
    #         ret, frame = vidcap.read()

    #         if ret:

    #             preds = model(frame)
    #             detected_objects = preds.pandas().xyxy[0]

    #             if len(detected_objects) > 0:
    #                 print(detected_objects)

    #                 for obj in detected_objects.itertuples():
    #                     process_predictions.predictions_visualization(obj, frame) 

    #             cv2.imshow("Frame", frame)

    #             if cv2.waitKey(1) & 0xFF==ord('q'):
    #                 break
    #         else:
    #             print("Error : no frame -- break")
    #             break