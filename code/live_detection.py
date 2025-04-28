import cv2
import numpy as np
import tensorflow as tf

from preprocess import Datasets

#change this to actual model
model = tf.keras.models.load_model() #insert .h5 file in parens

#unsure what to put for path
datasets = Datasets(, 1)

#helper method to preprocess images in the same way we did for training
def preprocess_frame(img):
    img = img / 255
    img = datasets.standardize(img)
    return img

#live video feed
live_feed = cv2.VideoCapture(0)

while live_feed.isOpened():
    return_val, frame_im = live_feed.read()
    if not return_val:
        break

    frame_im = preprocess_frame(frame_im)

    predictions = model.predict(frame_im)
    print("this is the prediction of the model:" + predictions)

    #assign label based on prediction
    label = 

    #assign color based on prediction
    color = 

    cv2.putText(frame_im, f"{label}", cv2.FONT_HERSHEY_PLAIN, 1, color)

    cv2.imshow("Live Detection", frame_im)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

live_feed.release()
cv2.destroyAllWindows()





    
