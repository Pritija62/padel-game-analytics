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
    str(BASE_DIR.parent / "output" / "output.mp4"), #saving output video in this path
    fourcc,
    fps,
    (width, height)
)
cv2.namedWindow("Output_tracked", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output_tracked", 1080, 720)

frame_number = 0
shot_events = []

forehand_count = 0
backhand_count = 0
shot_display_counter = 0

ret=True
while ret:

    ret, frame = cap.read()

    if not ret:
        break
    
    frame=cv2.resize(frame, (1080, 720))  
    # Tracking
    frame, detections = track_objects(frame)

    # Shot classification
    shot = classify_shot(frame_number, detections)

    if shot:
        print(shot)
        shot_events.append(shot)
        shot_display_counter = int(fps) 
        
        if shot["shot_type"] == "Forehand":
          forehand_count += 1 
        elif shot["shot_type"] == "Backhand":
          backhand_count += 1
    
    #  draw counter panel
    
    cv2.putText(frame, "Shot Analytics", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Forehands: {forehand_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Backhands: {backhand_count}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Total: {len(shot_events)}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Draw current shot type if recently detected
    if shot_display_counter > 0:
        shot_type = shot["shot_type"] if shot else "N/A"
        cv2.putText(frame, f"Shot: {shot_type}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        shot_display_counter -= 1
    # Display the frame
    cv2.imshow("Output_tracked", frame)
    frame = cv2.resize(frame, (width, height))
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save CSV
csv_path = BASE_DIR.parent / "output" / "shots.csv"

pd.DataFrame(shot_events).to_csv(csv_path, index=False)

# Save JSON
json_path = BASE_DIR.parent / "output" / "shots.json"



for shot in shot_events:

    if shot["shot_type"] == "Forehand":
        forehand_count += 1

    elif shot["shot_type"] == "Backhand":
        backhand_count += 1

print("----SHOT ANALYTICS----")
print("Total Forehands:", forehand_count)
print("Total Backhands:", backhand_count)

with open(json_path, "w") as f:
    json.dump(shot_events, f, indent=4)


print("Processing Complete")
print("CSV and JSON files saved")