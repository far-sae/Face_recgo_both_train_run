import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# Load the model
model = resnet18(weights=None)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)  # Ensure this matches your training setup
model.load_state_dict(torch.load("/Users/farazsaeed/Face-Auth/script/resnet18_model.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict the class from the webcam frame
def predict_from_frame(frame, model, transform):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():
        output = model(image)
        print(f"Model raw output: {output}")  # Print the raw output of the model
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the class labels (update these with your actual labels)
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
 # Replace with actual class names

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict the class
    predicted_class = predict_from_frame(frame, model, transform)

    # Debugging: Print the predicted class index
    print(f"Predicted class index: {predicted_class}")

    # Ensure the predicted class is within the valid range
    if 0 <= predicted_class < len(class_labels):
        label = class_labels[predicted_class]
    else:
        label = "Unknown"  # Handle any out-of-range predictions

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
