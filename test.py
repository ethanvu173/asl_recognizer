import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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


def extract_landmarks(detection_results):
    wrist = detection_results.hand_landmarks[0]
    coords = []

    for landmark in wrist:
        coords.extend([landmark.x-wrist[0].x,
                       landmark.y-wrist[0].y,
                       landmark.z-wrist[0].z])

    return coords


def detect_hands(img, detector):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    output = detector.detect(img_mp)
    if output.hand_landmarks:
        normalized = extract_landmarks(output)
        print(len(normalized))
        print(normalized[:12])
        print(output.hand_landmarks[:12])

    img = draw_landmarks_on_image(img, output)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    MODEL_PATH = "./hand_landmarker.task"

    # Set up the hand detection model
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)


    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    while success:
        frame = detect_hands(frame, detector)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pass
    # main()