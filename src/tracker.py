from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# Loading the YOLOv8m model
model = YOLO(str(BASE_DIR / "yolov8m.pt"))


class PlayerIDMapper:
    """Maps ByteTrack IDs to stable player IDs across frames."""
    def __init__(self, distance_threshold=100, appearance_threshold=30):
        self.bytetrack_to_player_id = {}  # Maps ByteTrack ID -> stable Player ID
        self.player_positions = {}  # Stores last known position of each player ID
        self.next_player_id = 1
        self.distance_threshold = distance_threshold
        self.appearance_threshold = appearance_threshold
    
    def get_player_id(self, bytetrack_id, bbox, frame_roi=None):
        """Get stable player ID for a ByteTrack detection."""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # If we've seen this ByteTrack ID before, return its mapped player ID
        if bytetrack_id in self.bytetrack_to_player_id:
            player_id = self.bytetrack_to_player_id[bytetrack_id]
            self.player_positions[player_id] = (center_x, center_y)
            return player_id
        
        # Find closest existing player by distance and appearance
        best_player_id = None
        best_distance = float('inf')
        
        for player_id, (last_x, last_y) in self.player_positions.items():
            distance = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)
            if distance < best_distance and distance < self.distance_threshold:
                best_distance = distance
                best_player_id = player_id
        
        # Assign new or existing player ID
        if best_player_id is None:
            player_id = self.next_player_id
            self.next_player_id += 1
        else:
            player_id = best_player_id
        
        # Map ByteTrack ID to player ID
        self.bytetrack_to_player_id[bytetrack_id] = player_id
        self.player_positions[player_id] = (center_x, center_y)
        
        return player_id


player_mapper = PlayerIDMapper(distance_threshold=120)


CONF_THRESHOLDS = {
    "person":        0.65,   # Higher threshold to filter out false positives
    "sports ball":   0.10,   # ball is tiny and hard, be lenient
    "tennis racket": 0.15,   # racket is also tricky
    }


def track_objects(frame):

    detections = []

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker=str(BASE_DIR / "bytetrack.yaml")
    )

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name not in CONF_THRESHOLDS:
                continue

            if conf < CONF_THRESHOLDS[class_name]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = int(box.id[0]) if box.id is not None else -1

            # PLAYER
            if class_name == "person":
                stable_player_id = player_mapper.get_player_id(
                    track_id,
                    (x1, y1, x2, y2)
                )

                color = (0, 255, 0)
                label = f"Player {stable_player_id}"

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

            # Store detection data
            detections.append({
                "class_name": class_name,
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "label": label
            })

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame, detections
    

