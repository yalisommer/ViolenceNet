import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from preprocess import Datasets
from skimage.transform import resize
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--video_src", type=str, required=True, help="Path to video or live feed to be processed")
# args = parser.parse_args()

# video_path = args.video_src

# LIVE = False
# if args.video_src.isdigit():
#     video_path = int(args.video_src)
#     LIVE = True

#change this to actual model
model = Sequential([
              ## Add layers here separated by commas.

            #   #idea: have two conv before pool, change size to be smaller than 7
            #   Conv2D(15, kernel_size=7, strides=1, activation='relu'), # 10 conv kernals each size 5x5 with stride 1
            #   MaxPool2D(2), # should I be naming these? 2 here means that the size of the downscaled pool is 2x2
            #   Dense(64, activation='relu'), # was 32
            #   Dropout(0.5),
            #   Flatten(),
            #   Dense(2, activation='softmax')
            #   # probably overfitting -> 2 dense and one conv

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
model.load_weights("checkpoints/your_model/your.e016-acc0.9152.weights.h5") #insert .h5 file in parens

#unsure what to put for path (but I know it should be like the data directory 
# bc the preprocess Datasets code uses it that way I'm j not sure if its formatted corretcly)
datasets = Datasets("../data", 1)

#helper method to preprocess images in the same way we did for training
def preprocess_frame(img):
    img = resize(img, (224, 224, 3), preserve_range=True)
    img = img / 255
    img = datasets.standardize(img)
    img = np.array(img, dtype=np.float32)
    return img

#video feed
live_feed = cv2.VideoCapture(0)

# if not LIVE:

while live_feed.isOpened():
    return_val, frame_im = live_feed.read()
    if not return_val:
        break

    display_frame = frame_im.copy()
    
    frame_im = preprocess_frame(frame_im)
    frame_im = np.expand_dims(frame_im, axis=0)  # add batch dimension
    prediction = model.predict(frame_im)

    #Unsure what this will output it is a tensorflow method
    prediction = model.predict(frame_im)
    violence_prob = prediction[0][1]
    print(violence_prob)

    color = (0, 0, 0)
    if violence_prob > 0.49:
        label = "VIOLENCE DETECTED"
        color = (0, 0, 255)
    else:
        label = ""

    cv2.putText(display_frame, label, (200, 400), cv2.FONT_HERSHEY_PLAIN, 8, color, 4)
    cv2.imshow("Live Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

live_feed.release()
cv2.destroyAllWindows()


    
