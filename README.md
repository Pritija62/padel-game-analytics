# AI-Based Padel Shot Analysis System

## Project Overview

This project is a prototype AI-powered padel analytics system developed for an AI/ML internship assignment.

The system processes a padel gameplay video and performs:

- Player detection
- Object tracking
- Stable player ID mapping
- Rule-based shot classification
- Gameplay analytics generation
- Structured CSV and JSON output export

The project demonstrates a complete computer vision pipeline using YOLOv8, ByteTrack, and OpenCV.

---

# Features

- Real-time player detection using YOLOv8
- Object tracking using ByteTrack
- Stable player ID assignment
- Rule-based shot classification prototype
- Output video generation with overlays
- CSV analytics export
- JSON analytics export
- Modular project structure

---

# Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- NumPy
- Pandas

---

# Project Structure

```txt
padel-project/
│
├── data/
│   └── sample_video.mp4
│
├── output/
│   ├── shots.csv
│   └── shots.json
│
├── src/
│   ├── main.py
│   ├── tracker.py
│   ├── shot_classifier.py
│   ├── yolov8m.pt
│   └── bytetrack.yaml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How It Works

## 1. Video Processing

The input gameplay video is read frame-by-frame using OpenCV.

---

## 2. Object Detection
Pretrained YOLOv8m weights from Ultralytics were used for object detection.
#### YOLOv8 detects:
- players
- sports racket (limited detection)
- sports ball (limited detection)

---

## 3. Object Tracking

ByteTrack is used to maintain tracking IDs across video frames.

---

## 4. Stable Player IDs

A custom player ID mapper was implemented to improve consistency of player identities throughout gameplay.

---

## 5. Shot Classification

A simplified rule-based shot classification system generates gameplay events such as:
- Forehand
- Backhand

The current implementation uses lightweight prototype logic to demonstrate the analytics pipeline.

---

## 6. Analytics Export

Detected shot events are exported into:

### CSV Output

```txt
frame,shot_type,player
120,Forehand,Player 1
240,Backhand,Player 1
```

### JSON Output

```json
[
    {
        "frame": 120,
        "shot_type": "Forehand",
        "player": "Player 1"
    }
]
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Pritija62/padel-game-analytics.git
cd padel-game-analytics
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run The Project

```bash
python src/main.py
```

---

# Output

The system generates:
- annotated output video
- shot analytics CSV
- shot analytics JSON

---

# Challenges Faced

Some challenges encountered during development:

- Small sports ball detection difficulty
- Fast object movement and motion blur
- Inconsistent racket visibility
- Maintaining stable player IDs
- Limited training data and project time constraints

---

# Future Improvements

Possible future enhancements:

- Custom-trained padel detection model
- Better ball tracking
- Bounce detection
- Pose estimation
- Deep learning-based shot classification
- Real-time dashboard visualization

---

# Demo Video

outputs and Demo video link:

https://drive.google.com/drive/u/0/folders/1-4FuBAsyl8SlXbIftlTAcSneZlRGcnLB

---

# Author

Pritija

---

# Notes

This project is a prototype implementation developed within a limited timeframe for internship evaluation purposes. The focus was on building a complete end-to-end analytics pipeline rather than achieving production-level sports analytics accuracy.
