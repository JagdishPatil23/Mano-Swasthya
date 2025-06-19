import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np

# Load Hugging Face model

MODEL_NAME = "dpaul93/face_emotions_image_detection-v2"  # Replace with your chosen model
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
TF_ENABLE_ONEDNN_OPTS=0
# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        try:
            # Extract face from frame
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Preprocess face for model
            inputs = feature_extractor(images=face_pil, return_tensors="pt")

            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax().item()

            # Get label from model
            label = model.config.id2label.get(predicted_class, "Unknown")

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    # Show video feed
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
