import cv2
import pandas as pd

def filter_predictions(df):
    return df[df['confidence'] >= 5]


def extract_detected_object_infos(row):
   index, data = row
   
   detected_object = {
        'xmin': int(data['xmin']),
        'ymin': int(data['ymin']),
        'xmax': int(data['xmax']),
        'ymax': int(data['ymax']),
        'confidence': data['confidence'],
        'label': data['name']
    }
   
   return detected_object


def draw_bouding_box_label_and_confidence(data, image):

    # Draw bouding_box
    cv2.rectangle(image, (data['xmin'], data['ymin']), (data['xmax'], data['ymax']), (0, 0, 255), 2)
    # Draw label and confidence 
    cv2.putText(image, f"{data['label']} - {data['confidence']:.2f}", (data['xmin'], data['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

