import cv2
import csv
import mediapipe as mp
from utils import extract_landmarks
from utils import draw_landmarks_on_image
from utils import init_hand_model


SAMPLES_PER_CLASS = 300
FILE_NAME = "output.csv"
MODEL_PATH = "../models/hand_landmarker.task"
EXIT_KEY = 27 # escape key
CAMERA_IDX = 0


# Load the hand landmarking model from MediaPipe
detector = init_hand_model(MODEL_PATH)
capture = cv2.VideoCapture(CAMERA_IDX)

with open(FILE_NAME, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    collecting = False
    label = ""
    count = 0

    success, frame = capture.read()
    while success:
        # Convert cv2 image (BGR by default) to RGB (what MediaPipe expects)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the cv2 image into a format MediaPipe can use
        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        output = detector.detect(frame_mp)
        key = cv2.waitKey(1) & 0xFF

        if key == EXIT_KEY:
            break
        # If a letter keystroke is detected, set the program to start collecting
        # the letter
        elif 65 <= key <= 90 or 97 <= key <= 122 and not collecting:
            collecting = True
            label = chr(key).upper()
            count = 0
            print(f"Collecting {label}")

        # If ready to collect data, get landmarks and write to the CSV file
        # with the label determined above
        if collecting and output.hand_landmarks:
            coords = extract_landmarks(output)
            writer.writerow([label] + coords)
            count += 1
            if count >= SAMPLES_PER_CLASS:
                collecting = False
                print(f"Done collecting {label}")

        if collecting:
            status = f"Collecting {label}: {count}"
        else:
            status = "Ready"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = draw_landmarks_on_image(frame, output)
        cv2.imshow("Press esc to quit", frame)

        success, frame = capture.read()
