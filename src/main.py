from pathlib import Path
import cv2
import pandas as pd
import json

from tracker import track_objects
from shot_classifier import classify_shot

BASE_DIR = Path(__file__).resolve().parent

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
    (1080, 720)
)
cv2.namedWindow("Output_tracked", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output_tracked", 1080, 720)

frame_number = 0
shot_events = []
ret=True
while ret:

    ret, frame = cap.read()

    if not ret:
        break
    
    
    frame=cv2.resize(frame, (1080, 720))  # Resize frame to 1080x720 for faster processing
    # Tracking
    frame, detections = track_objects(frame)

    # Shot classification
    shot = classify_shot(frame_number, detections)

    if shot:
        shot_events.append(shot)

        # Draw lebel for shot type
        cv2.putText(
                frame,
                shot["shot_type"],
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
    # Display the frame
    cv2.imshow("Output_tracked", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save CSV
csv_path = BASE_DIR.parent / "output" / "shots.csv"

pd.DataFrame(shot_events).to_csv(csv_path, index=False)


# Save JSON
json_path = BASE_DIR.parent / "output" / "shots.json"

with open(json_path, "w") as f:
    json.dump(shot_events, f, indent=4)


print("Processing Complete")
print("CSV and JSON files saved")