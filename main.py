import torch
import cv2

from drawBB import filter_predictions, draw_bounding_box_label_and_confidence

# Model Integration
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# Real-time capture of webcam video

vidcap = cv2.VideoCapture("http://192.168.1.19:8000/") # Create a variable that will hold the video capturing object.

if vidcap.isOpened():
    print("Cam open")

    while True:
        ret, frame = vidcap.read()
        if ret:

          # Inference
            preds = model(frame)
            detected_objects = preds.pandas().xyxy[0]
            filtered_objects = filter_predictions(detected_objects)

            if len(filtered_objects) > 0:
                print(filtered_objects)

                for obj in filtered_objects.itertuples():
                    draw_bounding_box_label_and_confidence(obj, frame) 

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break



# if vidcap.isOpened():
#     print("Camera is open")
    
#     while True:
#         ret, frame = vidcap.read()
#         print(f"ret: {ret}, frame: {frame}")

#         if frame is None:
#             print(" -- No captured frame : break -- ")
#         else:
#             # Inference
#             preds = model(frame)
#             detected_objects = preds.pandas().xyxy[0]
#             filtered_objects = filter_predictions(detected_objects)

#             if len(filtered_objects) > 0:
#                 print(filtered_objects)

#                 for obj in filtered_objects.itertuples():
#                     draw_bounding_box_label_and_confidence(obj, frame) 

#             # Display
#             cv2.imshow("Frame", frame)

#             if cv2.waitKey(1) & 0xFF==ord('q'):
#                 break
        
#             print("Closing camera")
#             vidcap.release()

# else:
#    print("Cannot open camera")
