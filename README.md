# Emotion Detection Using Facial Expressions

This project uses a deep learning model to detect emotions from facial expressions in real-time. It uses OpenCV for face detection and a pre-trained Keras model to classify emotions into one of the following categories: **Angry**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

## Features
- Real-time emotion detection from webcam feed.
- Uses a Haar Cascade Classifier for face detection.
- Classifies facial expressions into one of the five emotions: **Angry**, **Happy**, **Neutral**, **Sad**, **Surprise**.
- Displays the predicted emotion on the webcam feed.

## Requirements
To run this project, you'll need Python and the following libraries:

- OpenCV
- Keras
- TensorFlow
- NumPy

You can install the necessary dependencies by running the following command:

```bash
pip install opencv-python keras tensorflow numpy
```

### Dependencies:
- **OpenCV**: Used for capturing webcam feed and face detection.
- **Keras**: For loading the pre-trained emotion classification model.
- **NumPy**: For handling arrays and image preprocessing.

## Files:
- `test.py`: The main Python script to run the emotion detection.
- `emotion_recognition.keras`: The pre-trained Keras model used to classify emotions.
- `haarcascade_frontalface_default.xml`: A Haar Cascade Classifier XML file used for face detection.

## How to Run:
1. Clone the repository or download the project folder.
2. Install the required libraries.
3. Make sure the model file (`emotion_recognition.keras`) and the Haar Cascade XML (`haarcascade_frontalface_default.xml`) are in the correct locations.
4. Run the `test.py` script:

```bash
python test.py
```

The webcam feed will open, and the detected emotion will be displayed on the screen.

## Dataset:

This project uses the FER2013 dataset, which is a popular facial emotion dataset. It contains 35,887 48x48 grayscale images, each labeled with one of 7 emotions:

Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
Download the dataset from Kaggle and place it in the dataset/ folder.

## Model Architecture:

The model is built using the MobileNetV2 architecture, which is a lightweight convolutional neural network designed for mobile devices. It is trained on the FER2013 dataset and fine-tuned for facial emotion recognition.

## Key points:

MobileNetV2 is used due to its efficiency in handling real-time predictions.
The model is fine-tuned to handle 7 emotions.
The final layer uses a softmax activation function to classify images into one of the 7 emotion classes.
Testing the Model

Face Detection: OpenCV is used for detecting faces in real-time using Haar cascades.
Emotion Prediction: The model predicts the emotion on detected faces using the emotion_recognition.keras model.

