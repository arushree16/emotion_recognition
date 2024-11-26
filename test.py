from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load pre-trained model and face detection classifier
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./emotion_recognition.keras')

# Define class labels for emotion detection
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Adjust based on model output

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Start capturing video frames
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Region of interest (ROI) in grayscale
        roi_gray = gray[y:y + h, x:x + w]
        
        # Resize the ROI to 224x224 (model input size)
        roi_resized = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)

        if np.sum([roi_resized]) != 0:
            # Convert to float32, normalize, and expand dimensions to match model input shape
            roi = roi_resized.astype('float') / 255.0
            roi = img_to_array(roi)  # Convert to numpy array
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Convert grayscale to RGB (since the model expects 3 channels)
            if roi.shape[-1] == 1:  # If grayscale, convert to RGB
                roi = np.repeat(roi, 3, axis=-1)

            # Predict emotion
            preds = classifier.predict(roi)[0]
            print("\nprediction = ", preds)
            print("\nprediction shape = ", preds.shape)  # Debug the shape

            # Ensure the length of class_labels matches the output of the model
            if len(class_labels) > len(preds):
                print("Warning: The number of class labels is less than the model output.")

            # Get the predicted label
            label = class_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            
            # Position the label at the face's location
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # If no face is detected, display a message
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        print("\n\n")
    
    # Show the frame with the emotion label
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
