from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# Loading the YOLOv8m model
model = YOLO(str(BASE_DIR / "yolov8m.pt"))

video_path = BASE_DIR.parent / "data" / "sample_video.mp4"

cap = cv2.VideoCapture(str(video_path))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Creating video writer for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    str(BASE_DIR.parent / "output" / "output.mp4"),#saving output video in this path
    fourcc,
    fps,
    (width, height)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", width, height)



while True:

    ret, frame = cap.read()

    if not ret:
        break

    # TRACKING
    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        imgsz=640,
        verbose=False,
        tracker=str(BASE_DIR / "bytetrack.yaml"),
    )

    CONF_THRESHOLDS = {
    "person":        0.40,   # players are easy to detect, keep it strict
    "sports ball":   0.10,   # ball is tiny and hard, be lenient
    "tennis racket": 0.15,   # racket is also tricky
    }
    

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf   = float(box.conf[0])

            if class_name not in CONF_THRESHOLDS:
                continue

            if conf < CONF_THRESHOLDS[class_name]:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # TRACK ID
            track_id = int(box.id[0]) if box.id is not None else -1

            # PLAYER
            if class_name == "person":
                color = (0, 255, 0)
                label = f"Player {track_id}"

            # BALL
            elif class_name == "sports ball":
                color = (0, 0, 255)
                label = f"Ball {track_id}"

            # RACKET
            elif class_name == "tennis racket":
                color = (255, 0, 0)
                label = f"Racket {track_id}"

            else:
                continue

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # DRAW LABEL
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Output", frame)
    cap.grab()  

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()