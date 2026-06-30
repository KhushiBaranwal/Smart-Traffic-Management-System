# Smart Traffic Management System

A real-time, YOLOv8-based traffic monitoring and adaptive signal control system that detects vehicles, pedestrians, emergency vehicles, and mobility-impaired individuals to dynamically manage intersection signal timing — improving both traffic flow and accessibility.

## Overview

Traditional traffic signals run on fixed timers regardless of actual road conditions. This project uses real-time computer vision to detect what's actually happening at an intersection and adjusts signal timing accordingly — prioritizing emergency vehicles and giving extended crossing time to pedestrians who need it.

## Key Features

- **Real-time multi-class detection** — vehicles, pedestrians, emergency vehicles, and mobility-aid users (wheelchair/crutch) using a custom-trained YOLOv8 model
- **Adaptive signal control** — dynamic 5–15 second signal allocation based on live traffic density instead of fixed timers
- **Emergency vehicle prioritization** — automatically extends or holds green signal when an emergency vehicle is detected approaching
- **Accessibility-aware crossing logic** — extended pedestrian crossing windows for wheelchair and crutch users
- **Validated performance** — evaluated using confusion matrix analysis across all detection classes

## Model Performance

Trained over 50 epochs on a custom dataset, achieving:

| Metric | Score |
|---|---|
| Precision | 89% |
| Recall | 87% |
| mAP | 91% |

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV
- **Language:** Python
- **Dataset:** Custom-labeled dataset (Roboflow)

## Project Structure

```
YOLO_PROJECT/
├── simulation/
│   └── main.py          # Core detection + signal control logic
├── weights/
│   └── best.pt           # Trained YOLOv8 model weights
├── yolo26n.pt             # Base model checkpoint
├── yolov8n.pt              # Base model checkpoint
└── .gitignore
```

> Note: The training dataset is not included in this repository due to size. It was custom-labeled using [Roboflow](https://roboflow.com) and includes annotated classes for vehicles, pedestrians, emergency vehicles, and mobility-aid users.

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
