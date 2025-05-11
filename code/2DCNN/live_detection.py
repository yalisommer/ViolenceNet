import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from preprocess import Datasets
from skimage.transform import resize

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
model.load_weights('your.e016-acc0.9152.weights.h5') #this path can be changed to whatever weights 

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

#video feed
live_feed = cv2.VideoCapture(0)

while live_feed.isOpened():
    return_val, frame_im = live_feed.read()
    if not return_val:
        break

    display_frame = frame_im.copy()
    
    frame_im = preprocess_frame(frame_im)
    frame_im = np.expand_dims(frame_im, axis=0)  # add batch dimension
    prediction = model.predict(frame_im)

    prediction = model.predict(frame_im)
    violence_prob = prediction[0][1]
    print(violence_prob)

    color = (0, 0, 0)
    if violence_prob > 0.55:
        label = "VIOLENCE DETECTED"
        color = (0, 0, 255)
    else:
        label = ""

    cv2.putText(display_frame, label, (400, 800), cv2.FONT_HERSHEY_PLAIN, 8, color, 4)
    cv2.imshow("Live Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

live_feed.release()
cv2.destroyAllWindows()
