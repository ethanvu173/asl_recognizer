# ASL Alphabet Recognizer
A project to recognize the American Sign Language alphabet using hand gestures from a camera feed.

## Requirements
### Libraries
* OpenCV
* MediaPipe 0.10.0+
* NumPy
* scikit-learn
* Joblib
### Files
* **asl_alpha_classifier.pkl**<br>
  A provided classification model to predict the alphabet in ASL
* **hand_landmarker.task**<br>
MediaPipe's hand detection model. It is provided by Google and available [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

## Usage
### ASL Recognition
Run the main recognition program by executing **recognize.py**. Sign with the right hand only, and the predicted letter with confidence will appear. If a recognized letter is held for approximately 1 seconnd, it will be printed to the terminal. Sign "yes" in ASL to add a space, and sign "no" to delete a letter. Note that the program only recognizes the static variety of "no" (i.e. no movement in the sign).
### Hand Gesture Data Collection
To gather custom data to retrain the ASL recognition model, execute **data_collection.py**. Sign a letter and type the corresponding letter to begin recording data. Change the value of SAMPLES_PER_CLASS to gather more or less data. The data is written to a file named *output.csv*. 
### ASL Model Retraining
To retrain the ASL recognition model, ensure your custom data has been collected using data_collection.py. Exectute **train.py**. The model is outputted with the name *asl_alpha_classifier.pkl*.

## Roadmap
* Implement recognition for left hand

## Attributions
This project contains a method to draw the landmarks on a hand from Google's hand landmarker example code. It is available [here](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=TUfAcER1oUS6) and can be used under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). 
All other code has been made by myself.
