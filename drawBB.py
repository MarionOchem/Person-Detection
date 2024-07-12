import cv2
import pandas as pd

def filter_predictions(df):
    return df[(df['confidence'] >= 0.5) & (df['class'] == 0)]


def draw_bounding_box_label_and_confidence(data, image):
    # Draw bounding box
    cv2.rectangle(image, (int(data.xmin), int(data.ymin)), (int(data.xmax), int(data.ymax)), (0, 0, 255), 2)
    
    # Draw label and confidence 
    cv2.putText(image, f"{data.name} - {data.confidence:.2f}", (int(data.xmin), int(data.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
