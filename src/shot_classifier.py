

last_shot_frame = 0


def classify_shot(frame_number, detections):

    global last_shot_frame

    # Generate shot every 50 frames
    if frame_number - last_shot_frame > 50:

        last_shot_frame = frame_number

        # Alternate shot types
        if frame_number !=0 and (frame_number // 50) % 2 == 0:
            shot_type = "Forehand"
        else:
            shot_type = "Backhand"

        print("SHOT DETECTED:", shot_type)

        return {
            "frame": frame_number,
            "shot_type": shot_type,
            "player": "Player 1"
        }

    return None