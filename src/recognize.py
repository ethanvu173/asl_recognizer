import cv2
import joblib
import numpy as np
import mediapipe as mp
import utils
from utils import extract_landmarks
from utils import init_hand_model
from utils import draw_landmarks_on_image
from collections import deque


MODEL_PATH = "../models/hand_landmarker.task"
CAMERA_IDX = 0
EXIT_KEY = 27  # escape key
TRAJECTORY_LEN = 20
index_trajectory = deque(maxlen=TRAJECTORY_LEN)
pinky_trajectory = deque(maxlen=TRAJECTORY_LEN)
wrist_trajectory = deque(maxlen=TRAJECTORY_LEN)

# Set up video capture and hand landmarking model
classifier = joblib.load('../models/asl_alpha_classifier.pkl')
capture = cv2.VideoCapture(CAMERA_IDX)
detector = init_hand_model(MODEL_PATH)


prev_prediction = ''
phrase = ""
detect_count = 0
motion_sign = False

success, frame = capture.read()
while success:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    output = detector.detect(frame_mp)

    if output.hand_landmarks:
        for landmark in output.hand_landmarks:
            frame = draw_landmarks_on_image(frame, output)
            # Get index and pinky trajectories to detect letters J and Z
            wrist = landmark[0]
            i_tip = landmark[8]
            p_tip = landmark[20]
            index_trajectory.append((i_tip.x - wrist.x, i_tip.y - wrist.y))
            pinky_trajectory.append((p_tip.x - wrist.x, p_tip.y - wrist.y))
            wrist_trajectory.append((wrist.x, wrist.y))

        # Make predictions on hand landmarks.
        # Check motion-based signs first
        if utils.detect_yes(wrist_trajectory):
            prediction = 'YES'
            proba = 1
        elif utils.detect_z(index_trajectory):
            prediction = 'Z'
            proba = 1
        elif utils.detect_j(pinky_trajectory, index_trajectory):
            prediction = 'J'
            proba = 1
        else:
            # Get hand landmarks for static signs and make predictions
            coords = np.array(extract_landmarks(output)).reshape(1, -1)
            prediction = classifier.predict(coords)[0]
            proba = classifier.predict_proba(coords).max()

        # Requires >0.5 confidence to determine letter
        if proba > 0.5:
            cv2.putText(frame, f"{prediction}  {proba:.0%}",
                        (10, 50),cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 2)

            if prediction == prev_prediction:
                detect_count += 1
            else:
                detect_count = 0

            if (detect_count > 25 or
                    ((prediction in ['J', 'Z']) and detect_count > 10)):
                print("Detected: " + prediction)

                if prediction == 'NO':
                    phrase = phrase[:-1]
                else:
                    phrase += ' ' if prediction == 'YES' else prediction
                print(phrase)
                detect_count = 0
                motion_sign = False

            prev_prediction = prediction


    cv2.imshow("Press esc to quit", frame)
    success, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == EXIT_KEY:
        break

capture.release()
cv2.destroyAllWindows()
