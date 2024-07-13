''' This file configure the YOLOv5 model from PyTorch Hub API for this inference usage '''

import torch
import intel_extension_for_pytorch as ipex 

# operator pattern fusion to improve the efficiency of model execution. 

def initialize_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval() # Switches to evaludation mode / inference
    model.classes = [0]
    model.conf = 0.6
    model = ipex.optimize(model, dtype=torch.float32)

    if model is not None :
        return model
    else: 
        print("Error : Failed to load model")
        return None
    

# je n'ai pas utilisé de class pour définir le model prcq 
# je n'avais pas besoin de cette modularité : je n'ai qu'un model, 
# un detaset et un use case d'inférence