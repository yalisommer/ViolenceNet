import cv2
import numpy as np
from collections import deque
from keras.models import load_model

import os

# === CONFIGURABLE PARAMETERS ===
SEQUENCE_LENGTH = 16  
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  
CLASSES_LIST = ["NonViolence", "Violence"]  
MODEL_PATH = 'model-3dcnn9394.h5'  

cnn_model = load_model(MODEL_PATH)

def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):

    # Read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Store the predicted class in the video.
    predicted_class_name = ''

    nv_prob = 0
    v_prob = 0
    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = cnn_model.predict(np.expand_dims(frames_queue, axis = 0))[0]
            print("made a prediction")

            nv_prob = predicted_labels_probabilities[0]
            v_prob = predicted_labels_probabilities[1]


            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        cv2.putText(frame, "nv con: " + str(nv_prob), (5, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)
        cv2.putText(frame, "v con: " + str(v_prob), (5, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)

        # Write The frame into the disk using the VideoWriter
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

def predict_video(video_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)

    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Store the predicted class in the video.
    predicted_class_name = ''

    predicted_labels_and_confidences = []
    predicted_labels = []

    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = cnn_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            
            predicted_confidence = predicted_labels_probabilities[predicted_label]

            predicted_labels.append(predicted_class_name)

            predicted_labels_and_confidences.append((predicted_class_name, predicted_confidence))

            frames_queue.clear()
           
    video_reader.release()

    predicted_labels_array = np.array(predicted_labels)
    unique_labels, counts = np.unique(predicted_labels_array, return_counts=True)

    #thresholding at 50% (vote by majority) 

    #correction if only one label has been predicted
    if len(counts) == 1:
        if unique_labels[0] == "NonViolence":
            counts = np.array([counts[0], 0])
        else:
            counts = np.array([0, counts[0]])


    #thresholding
    if counts[1]/(counts[0] + counts[1]) > .5: #currently voting by majority of predictions can be changed
        final_prediction = "Violence" 
    else:
        final_prediction = "NonViolence"

    confidences = []
    for pair in predicted_labels_and_confidences:
        if pair[0] == final_prediction:
            confidences.append(pair[1])
        else:
            confidences.append((1 - pair[1]))

    confidences_array = np.array(confidences)
    final_confidence = np.mean(confidences_array)

    print("Predicted: " + final_prediction)
    print("Confidence: " + str(final_confidence))

    return final_prediction, final_confidence


#Specifying video to be predicted
# input_video_file_path = "Input_videos/control1.mp4"

# Perform Prediction on the Test Video.
# predict_frames(input_video_file_path, "Output_videos/dali_test_cons_vid.mp4", SEQUENCE_LENGTH)

# predict_video(input_video_file_path, SEQUENCE_LENGTH)


input_dir = "Input_videos"

#script to run experiment on the 25 violent videos
def predict_violent_vids():
    num_correct = 0
    num_false_negatives = 0
    to_review = 0
    review_threshold = .85
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4") and (filename.startswith("yt")): 
            input_path = os.path.join(input_dir, filename)
            print(f"Processing: {input_path}")
            prediction, confidence = predict_video(input_path, SEQUENCE_LENGTH)
            if prediction == "Violence":
                num_correct = num_correct + 1
            else:
                num_false_negatives = num_false_negatives + 1
                if confidence < review_threshold:
                    to_review = to_review + 1
    print("Outcomes on 25 violent videos:")
    print("Accuracy over 25 violent videos: " + str((num_correct/25)))
    print("False negatives: " + str(num_false_negatives))
    print("If threshold for nv confidence is " + str(review_threshold) + " else review: " + str(to_review) + " to be reviewed")
    print("In this case: " + str(num_correct) + " correct, " + str((num_false_negatives - to_review)) + " incorrect, " + str(to_review) + " to be reviewed")

def predict_nonviolent_vids():
    num_correct = 0
    num_false_positives = 0
    to_review = 0
    review_threshold = .85
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4") and (filename.startswith("control")): 
            input_path = os.path.join(input_dir, filename)
            print(f"Processing: {input_path}")
            prediction, confidence = predict_video(input_path, SEQUENCE_LENGTH)
            if prediction == "NonViolence":
                num_correct = num_correct + 1
                if confidence < review_threshold:
                    to_review = to_review + 1
            else:
                num_false_positives = num_false_positives + 1
    print("Outcomes on 25 nonviolent videos:")
    print("Accuracy over 25 nonviolent videos: " + str((num_correct/25)))
    print("False positives: " + str(num_false_positives))
    print("If threshold for nv confidence is " + str(review_threshold) + " else review: " + str(to_review) + " to be reviewed")
    print("In this case: " + str((num_correct - to_review)) + " correct, " + str(num_false_positives) + " incorrect, " + str(to_review) + " to be reviewed")


predict_violent_vids()
predict_nonviolent_vids()
