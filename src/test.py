import cv2
from ultralytics import YOLO

# Loading the YOLOv8n model
model= YOLO("yolov8n.pt") 

video_path = "../data/sample_video.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Creating video writer for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "../output/output.mp4",#saving output video in this path
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

    results = model(frame, conf=0.25)  

    # Looping through detected boxes and drawing them on the frame

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        #FOR PLAYER
        if class_name == "person":
            color = (0, 255, 0)
            label = "Player"

        # FOR BALL
        elif class_name == "sports ball":
            color = (0, 0, 255)
            label = "Ball"

        # FOR RACKET
        elif class_name == "tennis racket":
            color = (255, 0, 0)
            label = "Racket"

        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # show result
    cv2.imshow("Output", frame)

    # save frame
    out.write(frame)

    print([model.names[int(box.cls[0])] for box in results[0].boxes])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()