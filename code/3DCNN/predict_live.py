import cv2
import numpy as np
from collections import deque
from keras.models import load_model

#MAKE SURE TO RUN THIS FROM THE 3DCNN DIRECTORY

# === CONFIGURABLE PARAMETERS ===
SEQUENCE_LENGTH = 16  # Set according to how your model was trained
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  # Match your model input shape
CLASSES_LIST = ["NonViolence", "Violence"]  # Replace with your actual class names
MODEL_PATH = 'model_3dcnn.h5'  # Path to your saved .h5 model

# === Load the trained model ===
MoBiLSTM_model = load_model(MODEL_PATH)

# === Initialize video capture from webcam ===
video_reader = cv2.VideoCapture(0)

# === Initialize the frame queue ===
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

predicted_class_name = ''

nonviolence_confidence = 0
violence_confidence = 0

while True:
    ok, frame = video_reader.read()
    print("reading frame")
    if not ok:
        break

    # Resize and normalize the frame
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0

    # Append to frame queue
    frames_queue.append(normalized_frame)

    # Predict if we have enough frames
    if len(frames_queue) == SEQUENCE_LENGTH:
        print("\nGot enough frames, now predicting.\n")
        predicted_probs = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis=0), verbose=0)[0]
        nonviolence_confidence = predicted_probs[0]
        violence_confidence = predicted_probs[1]
        print(predicted_probs)

        #THRESHOLDING
        if nonviolence_confidence < .85:
            predicted_class_name = "Violence"
        else:
            predicted_class_name = "NonViolence"


        #JUST 50% PREDICTION
        # predicted_label = np.argmax(predicted_probs)
        # predicted_class_name = CLASSES_LIST[predicted_label]

    # Overlay prediction on the frame
    color = (0, 255, 0) if predicted_class_name != "Violence" else (0, 0, 255)
    cv2.putText(frame, predicted_class_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    cv2.putText(frame, "nonviolence confidence: " + str(nonviolence_confidence), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.putText(frame, "violence confidence: " + str(violence_confidence), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)


    # Show the frame
    cv2.imshow('Live Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_reader.release()
cv2.destroyAllWindows()