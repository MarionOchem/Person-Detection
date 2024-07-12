import cv2


def object_detection(model, image):
    preds = model(image)
    detected_objects = preds.pandas().xyxy[0]
    # print(detected_objects)

    return detected_objects


def predictions_visualization(data, image):
    # Draw bounding box
    cv2.rectangle(image, (int(data.xmin), int(data.ymin)), (int(data.xmax), int(data.ymax)), (0, 0, 255), 2)
    
    # Draw label and confidence 
    cv2.putText(image, f"{data.name} - {data.confidence:.2f}", (int(data.xmin), int(data.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
