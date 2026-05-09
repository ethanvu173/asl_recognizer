import cv2
import joblib
import numpy as np
import mediapipe as mp
from utils import extract_landmarks
from utils import init_hand_model
from utils import draw_landmarks_on_image


MODEL_PATH = "./hand_landmarker.task"
CAMERA_IDX = 0
EXIT_KEY = 'q'

# Set up video capture and hand landmarking model
classifier = joblib.load('asl_alpha_classifier.pkl')
capture = cv2.VideoCapture(CAMERA_IDX)
detector = init_hand_model(MODEL_PATH)

success, frame = capture.read()
while success:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    output = detector.detect(frame_mp)

    if output.hand_landmarks:
        for landmark in output.hand_landmarks:
            frame = draw_landmarks_on_image(frame, output)

        # Get and make predictions on hand landmarks
        coords = np.array(extract_landmarks(output)).reshape(1, -1)
        prediction = classifier.predict(coords)[0]
        proba = classifier.predict_proba(coords).max()

        # Requires >0.5 confidence to determine letter
        if proba > 0.5:
            cv2.putText(frame, f"{prediction}  {proba:.0%}",
                        (10, 50),cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 2)

    cv2.imshow("Press q to quit", frame)
    success, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
        break

capture.release()
cv2.destroyAllWindows()
