# SwiftScan
AI/ML Project
#### AI/ML Traffic Monitoring System

A real-time *vehicle detection*, *tracking*, and *license plate recognition* system, combining *YOLO* for object detection, *SORT* for tracking, and *EasyOCR* for license plate recognition to analyze traffic footage.

### Features

* Detect and track vehicles in real-time
* Recognize and extract license plate numbers
* Estimate speed and log over-speeding vehicles
* Region masking to focus on specific video areas

### Components

* `v2.py` – Main script for running the entire pipeline
* `sort.py` – Implements the SORT tracking algorithm
* `util.py` – Utility functions for speed calculation, coordinate mapping, etc.
* `util2.py` – Additional utilities
* `sort.cpython-*.pyc` & `util2.cpython-*.pyc` – Precompiled bytecode for faster imports
* `mask.png` – Region mask to specify the area of interest in the video
* `results.csv` – Logs overspeeding vehicle details (license plate and speed)
* `sample.mp4` – Sample traffic footage (Germany)

### How to Run

1. Install required dependencies (YOLO, OpenCV, EasyOCR)
2. Run the main script:

```bash
python v2.py
```

### Output

* Annotated video output with vehicle tracking and license plate recognition
* `results.csv` containing the details of overspeeding vehicles
