import torch
import cv2

from drawBB import filter_predictions, extract_detected_object_infos, draw_bouding_box_label_and_confidence

# Model Integration
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# Real-time capture of webcam video

vidcap = cv2.VideoCapture("http://192.168.1.19:8000/") # Create a variable that will hold the video capturing object.

if vidcap.isOpened():
    print("Camera is open")
   
    ret, frame = vidcap.read()

    # Inference
    preds = model(frame)
    detected_objects = preds.pandas().xyxy[0]
    # filtered_objects = filter_predictions(detected_objects)

    if detected_objects:
        print(detected_objects)
        while True:
            object_to_draw = [extract_detected_object_infos(row) for row in detected_objects.iterrows()]
            for obj in object_to_draw:
                draw_bouding_box_label_and_confidence(obj, frame)

            # Display
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    else:
        print("No predictions was made")
    
    print("Closing camera")
    vidcap.release()

else:
   print("Cannot open camera")
