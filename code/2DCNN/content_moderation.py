import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from preprocess import Datasets
from skimage.transform import resize

#This is our most successful 2D CNN (trained from scratch)
model = Sequential([
              Conv2D(32, 3, 1, activation='relu', padding='same'),
              BatchNormalization(),
              Conv2D(32, 3, 1, activation='relu', padding='same'),
              BatchNormalization(),
              MaxPool2D(2),

              Conv2D(64, 3, 1, activation='relu', padding='same'),
              BatchNormalization(),
              Conv2D(64, 3, 1, activation='relu', padding='same'),
              BatchNormalization(),
              MaxPool2D(2),

              Flatten(),
              Dense(86, activation='relu'),
              Dropout(0.4),
              Dense(2, activation='softmax')
        ])
model.build()
model.load_weights('your.e016-acc0.9152.weights.h5') #this path can be changed to whatever weights but architecture must be
#changed accordingly

#RESNET BASED MODEL
# input_shape = (224, 224, 3)
# base_input = tf.keras.layers.Input(shape=input_shape)
# resnet_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=base_input)
# x = resnet_base.output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Converts (7, 7, 2048) to (2048,)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(2, activation='softmax')(x)

# model = tf.keras.models.Model(inputs=base_input, outputs=output)

# model.build()
# model.load_weights('your.e045-acc0.9332.weights.h5')

datasets = Datasets('../../data', 1)

#helper method to preprocess images in the same way we did for training
def preprocess_frame(img):
    img = resize(img, (224, 224, 3), preserve_range=True)
    img = img / 255
    img = datasets.standardize(img)
    img = np.array(img, dtype=np.float32)
    return img

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
