import numpy as np
from ultralytics import YOLO
import cv2
from sort import Sort
import cvzone
import time
import csv
from util2 import *

# Constants and initialization
mot_tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.4)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("best.pt")
cap = cv2.VideoCapture("sample.mp4")
cy1, cy2, offset = 503, 550, 6

vh_down = {}
results = set()  # Use a set to store unique license plate texts
mask = cv2.imread("mask.png")
mask_resized = cv2.resize(mask, (1280, 720))
license_plate_detection_interval = 10
prev_time = time.time()
frame_count = 0

# Function to write results to a CSV file
def write_results_to_csv(results, file_name='results.csv'):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['License Plate Text'])
        for license_plate_text in results:
            writer.writerow([license_plate_text])

def object_detection(frame, model):
    mask_resized = cv2.resize(mask, (1280, 720))
    img_region = cv2.bitwise_and(frame, mask_resized)
    detections = model(img_region)[0]
    detection_list = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        detection_list.append([x1, y1, x2, y2, score, class_id])
    return detection_list

def draw_bounding_boxes(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Main loop for processing frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Object detection
        detections = object_detection(frame, coco_model)

        # Object tracking
        tracked_objects = mot_tracker.update(np.array(detections))

        # Draw bounding boxes around cars with their IDs
        draw_bounding_boxes(frame, tracked_objects)

        # Speed calculation
        curr_time = time.time()
        # speeds, overspeed_ids = calculate_speed(tracked_objects, vh_down, curr_time, cy1, cy2, offset, 47)

        # FPS calculation
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # License plate detection
        if frame_count % 10 == 0:
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, tracked_objects)

                if car_id != -1:
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read licence plate
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results.add(license_plate_text)
                        print(f"Detected license plate text: {license_plate_text}")

        frame_count += 1

        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()

# Write the results to a CSV file after processing
write_results_to_csv(results)