import cv2


def capture_video(url):
    vidcap = cv2.VideoCapture(url)
    if vidcap.isOpened():
        print("Camera is open")
        return vidcap
    else:
        print("Error : Failed to open camera")
        return None