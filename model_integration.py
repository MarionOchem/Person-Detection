import torch


def initialize_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.classes = [0]
    model.conf = 0.6

    if model is not None :
        return model
    else: 
        print("Error : Failed to load model")
        return None