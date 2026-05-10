# Pritija

previous_ball_position = None


def classify_shot(frame_number, detections):

    global previous_ball_position

    ball_position = None

    # Find ball
    for detection in detections:

        if detection["class_name"] == "sports ball":

            x1, y1, x2, y2 = detection["bbox"]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            ball_position = (center_x, center_y)

            break

    # No ball detected
    if ball_position is None:
        return None

    # First frame case
    if previous_ball_position is None:
        previous_ball_position = ball_position
        return None

    prev_x, prev_y = previous_ball_position
    curr_x, curr_y = ball_position

    # Ball movement
    dx = curr_x - prev_x
    dy = curr_y - prev_y

    shot_type = None

    # SIMPLE RULES

    # Moving strongly right
    if dx > 40:
        shot_type = "Forehand"

    # Moving strongly left
    elif dx < -40:
        shot_type = "Backhand"

    # Moving strongly downward
    elif dy > 50:
        shot_type = "Smash"

    previous_ball_position = ball_position

    if shot_type:

        return {
            "frame": frame_number,
            "shot_type": shot_type,
            "ball_position": ball_position
        }

    return None