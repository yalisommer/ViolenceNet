import cv2
import numpy as np
import time
from collections import deque
from keras.models import load_model
from ultralytics import YOLO

# === CONFIGURATION ===
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["NonViolence", "Violence"]
CNN_MODEL_PATH = 'model_3dcnn_global_.94.h5'
YOLO_MODEL_PATH = '../../yolo/my_model2.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.6
FPS_SMOOTHING = 10

# === Load Models ===
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
labels = yolo_model.names

print("Loading 3D CNN model...")
cnn_model = load_model(CNN_MODEL_PATH)

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
frame_queue = deque(maxlen=SEQUENCE_LENGTH)
violence_score_queue = deque([1]*8, maxlen=8)

# === FPS Tracking ===
fps_buffer = deque(maxlen=FPS_SMOOTHING)

predicted_class_name = ''
nonviolence_confidence = 0.0
violence_confidence = 0.0

# === Font Settings ===
font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_scale = 2.5
font_thickness = 4

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: could not read frame.")
        break

    display_frame = frame.copy()

    # === YOLO Inference ===
    yolo_results = yolo_model(frame, verbose=False)[0]
    detections = yolo_results.boxes

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        conf = detections[i].conf.item()
        classidx = int(detections[i].cls.item())
        label = labels[classidx]

        if conf > YOLO_CONFIDENCE_THRESHOLD:
            color = (0, 255, 255)
            cv2.rectangle(display_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label_text = f"{label}: {int(conf * 100)}%"
            cv2.putText(display_frame, label_text, (xyxy[0], max(30, xyxy[1] - 20)),
                        font_face, font_scale, color, font_thickness)

    # === CNN Prediction Prep ===
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255.0
    frame_queue.append(normalized_frame)

    # === CNN Inference ===
    if len(frame_queue) == SEQUENCE_LENGTH:
        prediction = cnn_model.predict(np.expand_dims(frame_queue, axis=0), verbose=0)[0]
        nonviolence_confidence = prediction[0]
        violence_confidence = prediction[1]
        violence_score_queue.append(nonviolence_confidence)

        if (violence_score_queue[-1] < 0.92 and violence_score_queue[-2] < 0.92) or violence_score_queue[-1] < 0.6:
            predicted_class_name = "Violence"
            cv2.putText(
                display_frame,
                "VIOLENCE DETECTED",
                (
                    int((display_frame.shape[1] - cv2.getTextSize("VIOLENCE DETECTED", font_face, font_scale + 2, font_thickness + 1)[0][0]) / 2),
                    int(display_frame.shape[0] * 0.72)
                ),
                font_face,
                font_scale + 2,
                (0, 0, 255),
                font_thickness + 1
            )
        else:
            predicted_class_name = "NonViolence"

    # === Overlay CNN Info ===
    color = (0, 255, 0) if predicted_class_name != "Violence" else (0, 0, 255)

    cv2.putText(display_frame, f"Prediction: {predicted_class_name}", (10, 100),
                font_face, font_scale + 0.5, color, font_thickness + 1)

    cv2.putText(display_frame, f"NonViolence: {nonviolence_confidence:.6f}", (10, 180),
                font_face, font_scale, (0, 255, 0), font_thickness)

    cv2.putText(display_frame, f"Violence: {violence_confidence:.6f}", (10, 260),
                font_face, font_scale, (0, 0, 255), font_thickness)

    # === FPS Overlay ===
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_buffer.append(fps)
    avg_fps = np.mean(fps_buffer)
    cv2.putText(display_frame, f"FPS: {avg_fps:.2f}", (10, 40),
                font_face, font_scale, (255, 255, 0), font_thickness)

    # === Display Result ===
    cv2.imshow("YOLO + CNN Output", display_frame)

    # === Exit on Q ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
