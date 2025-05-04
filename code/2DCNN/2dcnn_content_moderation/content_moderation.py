import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
from preprocess import Datasets
from skimage.transform import resize
from live_detection import preprocess_frame

model = Sequential([
              ## Add layers here separated by commas.

              #idea: have two conv before pool, change size to be smaller than 7
              Conv2D(15, kernel_size=7, strides=1, activation='relu'), # 10 conv kernals each size 5x5 with stride 1
              MaxPool2D(2), # should I be naming these? 2 here means that the size of the downscaled pool is 2x2
              Dense(64, activation='relu'), # was 32
              Dropout(0.5),
              Flatten(),
              Dense(2, activation='softmax')
              # probably overfitting -> 2 dense and one conv
        ])
model.build()
model.load_weights("/Users/yalisommer/Desktop/Schoolwork/CS/CS1430/ViolenceNet-Darpli/code/checkpoints/your_model/042825-151251/your.e048-acc0.8863.weights.h5") #insert .h5 file in parens

#unsure what to put for path (but I know it should be like the data directory 
# bc the preprocess Datasets code uses it that way I'm j not sure if its formatted corretcly)
datasets = Datasets("/Users/yalisommer/Desktop/Schoolwork/CS/CS1430/ViolenceNet-Darpli/data", 1)

#NEED TO decide how to feed the video in here
video_path = ""

video_feed = cv2.VideoCapture(video_path)

frames_per_sec = video_feed.get(cv2.CAP_PROP_FPS)

total_frames = int(video_feed.get(video_feed.CAP_PROP_FRAME_COUNT))

interval = int(frames_per_sec * 1)

time_stamps = []

frame_idx = 0
print("Checking " + video_path + "for violent content")
while video_feed.isOpened():
    return_val, frame_im = video_feed.read()
    if not return_val:
        break

    #every number of frames per sec (so 1 frame per sec)
    if frame_idx % interval == 0:
        frame_im = preprocess_frame(frame_im)
        prediction = model.predict(frame_im)
        violence_prob = prediction[0][1]
        print(violence_prob)

        if violence_prob > 0.5:
            #get the second of the violence
            time_stamp = frame_idx/frames_per_sec
            print("Violence detected at:" + time_stamp)
            time_stamps.append((time_stamp, violence_prob))

    frame_idx += 1

video_feed.release()


    

