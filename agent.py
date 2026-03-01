import os
# --- THE FIX: Force TensorFlow to use the Keras 2 engine ---
# This MUST be at the very top, before we import tensorflow!
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the Brain (No custom hacks needed anymore!)
print("Waking up the Vision Agent...")
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# 2. Open the Eyes (Webcam)
camera = cv2.VideoCapture(0)

print("Agent is now watching the line. Press 'q' to stop.")

while True:
    # Grab a frame from the camera
    success, image = camera.read()
    if not success:
        print("Failed to grab camera frame. Check if another app is using your webcam!")
        break

    # 3. Prepare the image for the AI
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # 4. Predict the Contents
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # --- 5. THE AGENT LOGIC (FSSAI RULES) ---
    # Clean up the class name (removes the "0 ", "1 " prefix from labels.txt safely)
    clean_class = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name
    
    # Check if the AI sees Contamination with high confidence
    if "Contaminated" in clean_class and confidence_score > 0.70:
        cv2.putText(image, "CRITICAL: CONTAMINATION DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"ALARM Triggered! Soil/Stone detected (Confidence: {np.round(confidence_score * 100)}%)")
    else:
        cv2.putText(image, f"Line Clear: {clean_class} ({np.round(confidence_score * 100)}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the camera feed
    cv2.imshow("FSSAI Vision Agent MVP", image)

    # Listen for the 'q' key to stop the program
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
