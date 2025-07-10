import string
import easyocr
import numpy as np
import cv2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}



def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None



def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def object_detection(frame, model):
    """
    Perform object detection on a frame using the provided YOLO model.

    Parameters:
        frame: numpy array
            The input frame.
        model: YOLO
            The YOLO model for object detection.

    Returns:
        detections: list
            List of detected objects in the frame.
    """
    mask_resized = cv2.resize(mask, (1280, 720))
    img_region = cv2.bitwise_and(frame, mask_resized)
    detections = model(img_region)[0]
    detection_list = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        detection_list.append([x1, y1, x2, y2, score, class_id])
    return detection_list

def object_tracking(detections, tracker):
    """
    Perform object tracking on the detected objects using the provided SORT tracker.

    Parameters:
        detections: list
            List of detected objects.
        tracker: Sort
            The SORT tracker for object tracking.

    Returns:
        tracked_objects: list
            List of tracked objects with their IDs and bounding boxes.
    """
    tracked_objects = tracker.update(np.array(detections))
    tracked_list = []
    for result in tracked_objects:
        x1, y1, x2, y2, ID = result
        tracked_list.append(( x1, y1, x2, y2,ID))
    return tracked_list

def calculate_speed(tracked_objects, vh_down, curr_time, cy1, cy2, offset, pixel_distance):
    """
    Calculate the speed of tracked vehicles and check for overspeed.

    Parameters:
        tracked_objects: list
            List of tracked objects with their IDs and bounding boxes.
        vh_down: dict
            Dictionary to store the downward crossing times of vehicles.
        curr_time: float
            Current time in seconds.
        cy1: int
            Y-coordinate of the first line for speed calculation.
        cy2: int
            Y-coordinate of the second line for speed calculation.
        offset: int
            Offset for y-coordinate comparison.
        pixel_distance: int
            Pixel distance corresponding to a known real-world distance.

    Returns:
        speeds: dict
            Dictionary containing vehicle IDs and their corresponding speeds.
        overspeed_ids: list
            List of IDs of vehicles detected to be overspeeding.
    """
    speeds = {}
    overspeed_ids = []
    for result in tracked_objects:
        (x1, y1, x2, y2, ID) = result
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if cy1 - offset < cy < cy1 + offset:
            vh_down[ID] = curr_time

        if ID in vh_down:
            if cy2 - offset < cy < cy2 + offset:
                elapsed_time = curr_time - vh_down[ID]
                if ID not in speeds:
                    pixel_distance = 47  # meters (considering the given value)
                    speed_ms = (pixel_distance * 1.5) / elapsed_time
                    speeds[ID] = speed_ms
                    if speed_ms > 15:
                        overspeed_ids.append(ID)

    return speeds, overspeed_ids

def calculate_fps(prev_time, curr_time):
    """
    Calculate frames per second (FPS) based on previous and current time.

    Parameters:
        prev_time: float
            Previous time in seconds.
        curr_time: float
            Current time in seconds.

    Returns:
        fps: float
            Frames per second (FPS).
    """
    fps = 1 / (curr_time - prev_time)
    return fps

def license_plate_detection(frame, license_plate_model):
    """
    Perform license plate detection on a frame using the provided YOLO model.

    Parameters:
        frame: numpy array
            The input frame.
        license_plate_model: YOLO
            The YOLO model for license plate detection.

    Returns:
        license_plates: list
            List of detected license plates along with their bounding boxes.
    """
    license_plates = []
    license_plate_boxes = license_plate_model(frame)[0]
    for box in license_plate_boxes.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        license_plates.append((x1, y1, x2, y2, score,class_id))
    return license_plates

