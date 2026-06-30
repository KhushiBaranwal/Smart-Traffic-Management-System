# Smart Traffic Management System

A real-time, YOLOv8-based intelligent traffic management system that detects transportation objects using a custom-trained object detection model and dynamically controls traffic signals based on traffic density, emergency vehicle priority, and pedestrian accessibility.

## Overview

Traditional traffic signals run on fixed timers regardless of actual road conditions. This project uses real-time computer vision to detect what's actually happening at an intersection and adjusts signal timing accordingly — prioritizing emergency vehicles and giving extended crossing time to pedestrians who need it.

## Key Features

- **Real-time multi-class detection** — vehicles, pedestrians, emergency vehicles, and mobility-aid users (wheelchair/crutch) using a custom-trained YOLOv8 model
- **Adaptive signal control** — dynamic 5–15 second signal allocation based on live traffic density instead of fixed timers
- **Emergency vehicle prioritization** — automatically extends or holds green signal when an emergency vehicle is detected approaching
- **Accessibility-aware crossing logic** — extended pedestrian crossing windows for wheelchair and crutch users
- **Validated performance** — evaluated using confusion matrix analysis across all detection classes

## Model Performance

Model trained using YOLOv8s for 50 epochs with an image size of 640×640., achieving:

| Metric | Value |
|---------|------:|
| Precision | 89% |
| Recall | 87% |
| mAP@0.5 | 91% |
| Epochs | 50 |
| Image Size | 640×640 |
| Model | YOLOv8s |

## Detection Strategy

The YOLOv8 model was trained on a custom transportation dataset containing 23 object classes.

For traffic signal control, detected objects are grouped into four logical categories.

| YOLO Classes | Traffic Category |
|--------------|------------------|
| Car, Bus, Truck, etc. | Vehicle |
| Ambulance, Fire Truck | Emergency Vehicle |
| Person | Pedestrian |
| Wheelchair, Crutches | Disabled Person |

This grouping enables efficient traffic signal decision-making while preserving detailed object detection.

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV
- **Language:** Python
- **Dataset:** Custom-labeled dataset (Roboflow)

## Project Structure

```
YOLO_PROJECT/
├── simulation/
│   └── main.py
├── weights/
│   └── best.pt
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

> Note:
> ## Dataset
 The dataset is not included due to its large size.

The model was trained on a custom transportation dataset exported from Roboflow consisting of 23 object classes.

For traffic signal decision-making, these classes are grouped into four logical categories during inference.

## How It Works

1. Live video feed is processed frame-by-frame through the trained YOLOv8 model
2. Detected objects are classified and counted per lane
3. A signal control algorithm calculates optimal green-light duration (5–15s) based on detected traffic density
4. If an emergency vehicle is detected, the system overrides normal logic to hold or extend the green signal on that lane
5. If a mobility-aid user is detected at a crossing, pedestrian signal duration is automatically extended

## Setup

```bash
# Clone the repository
git clone https://github.com/KhushiBaranwal/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System

# Create virtual environment
python -m venv yoloenv
yoloenv\Scripts\activate      # Windows
# source yoloenv/bin/activate  # macOS/Linux

# Install dependencies
pip install ultralytics opencv-python numpy

# Run
python simulation/main.py
```

## Future Improvements

- Multi-intersection coordination for corridor-level traffic optimization
- Edge deployment (Jetson Nano / Raspberry Pi) for real-world testing
- Integration with live city traffic camera feeds

## Author

**Khushi Baranwal**
B.Tech ECE, Jaypee Institute of Information Technology, Noida
[GitHub](https://github.com/KhushiBaranwal) · [LinkedIn](https://linkedin.com/in/khushi-baranwal)
