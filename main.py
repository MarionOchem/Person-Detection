import torch
import cv2

# Model Integration
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# Real-time capture of webcam video

vidcap = cv2.VideoCapture("http://192.168.1.19:8000/") # Create a variable that will hold the video capturing object.

if vidcap.isOpened():
    print("Camera is open")
   
    ret, frame = vidcap.read()

    # Inference
    result = model(frame)
    # Log coordinates results 
    # Extract coordinates results
    coor = result.pandas().xyxy[0]
    print(coor)
    if ret:
        while True:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    else:
        print("Failed to capture frame")
    
    vidcap.release()

else:
   print("Cannot open camera")

# Inference
# results = model(im)





# Draw result on video output 
