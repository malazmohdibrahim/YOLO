# YOLO Object Detection & Counter — Streamlit

A simple Python application that uses YOLO to detect people and objects from an uploaded image.

## Features

- Upload JPG, JPEG, PNG, or WEBP images
- YOLO object detection
- Adjustable confidence threshold
- Draw bounding boxes around detections
- Total number of detected objects
- Number of people detected
- Number of different object types
- Count of each object category
- Percentage of each object category
- Most common object type
- People percentage
- Detailed confidence table
- CSV report download

## Installation

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The first run downloads the `yolov8n.pt` pretrained YOLO model.

## Important note

The default YOLOv8n model is trained on the COCO dataset and can recognize common classes such as:

- person
- car
- bicycle
- motorcycle
- bus
- truck
- dog
- cat
- chair
- bottle
- laptop
- cell phone
- and other common object classes

The application counts detections (bounding boxes). If the same physical object is detected twice, it can appear twice in the count.
