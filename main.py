import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0,0],
    [1280 //2, 0],
    [1250 // 2, 720],
    [0, 720]
])

#webcam resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args= parser.parse_args()
    return args

def main():
    #webcam resolution
    args = parse_arguments()
    width, height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    #our model
    #model = YOLO("yolov8l.pt")
    model = YOLO("/Users/yaseminozkut/Desktop/best5.pt")
    #model.predict(source="0", show=True, conf=0.5)
    
    #box annotation apperiance
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone = sv.PolygonZone(
        polygon=ZONE_POLYGON, 
        frame_resolution_wh=tuple(args.webcam_resolution)
    )
    
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.blue(),
        thickness=3,
        text_thickness=4,
        text_scale=2
    )
    while True:
        ret, frame = cap.read() 

        result = model(frame, agnostic_nms=True, conf=0.5)[0] #agnostic_nms disables multi detection
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id!= 3] # eliminating the person class
        #detections = detections[detections.class_id== 0] # only counting the person class
        #detections = detections[detections.class_id== 41] # only counting the cup class
        #detections = detections[detections.class_id.isin([0, 41])] # Select "person" and "cup" classes
        #detections = detections[pd.Series(detections.class_id).isin([41, 42, 43, 44, 46])] #cup, spoon, fork, knife, banana

        #box labels
        labels = []

        for detection in detections:
            confidence = detection[2]
            class_id = detection[3]
            
            label = f"{model.model.names[class_id]} {confidence:0.2f}"
            labels.append(label)
        
        #adding the frame
        frame = box_annotator.annotate(scene=frame,  detections=detections, labels=labels)
        
        cv2.imshow("yolov8", frame)

        if(cv2.waitKey(30) == 27): #30 represents the ms, 27 represents esc key in ascii
            break
if __name__ == "__main__":
    main()