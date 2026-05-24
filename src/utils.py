import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


TRAJECTORY_LEN = 20

# Counts the amount of times a landmark reverses in the x-direction
def count_x_reversals(traj):
    points = list(traj)
    reversals = 0

    for i in range(2, len(points)):
        dx_prev = points[i-1][0] - points[i-2][0]
        dx_curr = points[i][0] - points[i-1][0]

        # If the signs do not match, there must be a reversal
        if dx_prev * dx_curr < -0.001:
            reversals += 1

    return reversals

# Counts the amount of times a landmark reverses iin the y-direction
def count_y_reversals(traj):
    points = list(traj)
    reversals = 0

    for i in range(2, len(points)):
        dy_prev = points[i-1][1] - points[i-2][1]
        dy_curr = points[i][1] - points[i-1][1]

        if dy_prev * dy_curr < 0.001:
            reversals += 1

    return reversals

def detect_yes(w_traj):
    if len(w_traj) < TRAJECTORY_LEN:
        return False

    points = list(w_traj)

    # To detect the nodding motion, check for at least 2 reversals
    if count_y_reversals(w_traj) < 2:
        return False

    # Check for significant vertical range
    y_list = [p[1] for p in points]
    if max(y_list) - min(y_list) < 0.06:
        return False

    # Check for small horizontal movement
    x_list = [p[0] for p in points]
    if max(x_list) - min(x_list) > 0.08:
        return False

    return True

# Detects the letter J using pinky and index finger trajectories
def detect_j(p_traj, i_traj):
    if len(p_traj) < TRAJECTORY_LEN:
        return False

    p_points = list(p_traj)
    i_points = list(i_traj)

    # Check that pinky moves significantly more than index
    p_list = [p[1] for p in p_points]
    i_list = [p[1] for p in i_points]
    p_range = max(p_list) - min(p_list)
    i_range = max(i_list) - min(i_list)
    if i_range > p_range * 0.6:
        return False

    dy = p_points[-1][1] - p_points[0][1]
    if dy < 0.05:
        return False

    x_list = [p[0] for p in p_points]
    if max(x_list) - min(x_list) < 0.03:
        return False

    return True

# Detects the letter Z based on index finger trajectories
def detect_z(traj):
    if len(traj) < TRAJECTORY_LEN:
        return False

    points = list(traj)

    if count_x_reversals(traj) != 2:
        return False

    x_list = [p[0] for p in points]
    if max(x_list) - min(x_list) < 0.08:
        return False

    return True

# Creates a hand landmarking model using the model at the location path
def init_hand_model(path):
    base_options = python.BaseOptions(model_asset_path=path)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    return detector

# Normalizes the hand landmarks with respect to the wrist landmark
def extract_landmarks(detection_results):
    wrist = detection_results.hand_landmarks[0]
    coords = []

    for landmark in wrist:
        coords.extend([landmark.x-wrist[0].x,
                       landmark.y-wrist[0].y,
                       landmark.z-wrist[0].z])

    return coords

# From the Google MediaPipe example for hand landmarks
# License: https://www.apache.org/licenses/LICENSE-2.0
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    vision.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks,
      vision.HandLandmarksConnections.HAND_CONNECTIONS,
      vision.drawing_styles.get_default_hand_landmarks_style(),
      vision.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - 10

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                1, (0,255,0), 1, cv2.LINE_AA)

  return annotated_image