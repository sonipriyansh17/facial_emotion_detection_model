import cv2
import numpy as np 
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from tensorflow.python.keras.models import model_from_json  # Use tensorflow.keras instead of keras

# Load the model architecture from JSON file
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load model weights
model.load_weights("facialemotionmodel.h5")  # Use the correct .h5 file for weights

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image
def extract_feature(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize the pixel values

# Start the webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face

        # Resize the face to match the input shape of the model
        face = cv2.resize(face, (48, 48))
        img = extract_feature(face)

        # Predict the emotion
        pred = model.predict(img)
        pred_label = labels[pred.argmax()]

        # Display the emotion label on the frame
        cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Facial Emotion Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
