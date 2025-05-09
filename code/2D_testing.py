from keras.models import load_model
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization


# import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from preprocess import Datasets
from skimage.transform import resize

from PIL import Image


# import hyperparameters as hp


CLASSES_LIST = ["NonViolence", "Violence"]  
MODEL_PATH = 'your.e016-acc0.9152.weights.h5' 

datasets = Datasets('../data', 1)

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
model.load_weights(MODEL_PATH)


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
# model.load_weights(MODEL_PATH)

#cnn_model = load_model(MODEL_PATH)

true_positives = 0 #violence predicted as violence
false_positives = 0 #nonviolence predicted as violence
true_negatives = 0 #nonviolence predicted as nonviolence
false_negatives = 0 #violence predicted as nonviolence

def preprocess_frame(img):
    img = resize(img, (224, 224, 3), preserve_range=True)
    img = img / 255
    img = datasets.standardize(img)
    img = np.array(img, dtype=np.float32)
    return img

nv_input_dir = '../data/valid/NV'

num_nv_correct = 0
num_nv = 0

for filename in os.listdir(nv_input_dir):
    if filename.endswith(".jpg") :  
        input_path = os.path.join(nv_input_dir, filename)
        print(f"Processing: {input_path}")

        img = Image.open(input_path)
        img = np.array(img)
        img = preprocess_frame(img)
        img = np.expand_dims(img, axis=0)

        predicted_labels_probs = model.predict(img)[0]
        predicted_label_index = np.argmax(predicted_labels_probs)
        predicted_class_name = CLASSES_LIST[predicted_label_index]
        num_nv = num_nv + 1
        if predicted_class_name == "NonViolence":
            num_nv_correct = num_nv_correct + 1
            true_negatives = true_negatives + 1
        else:
            false_positives = false_positives + 1

v_input_dir = '../data/valid/V'

num_v_correct = 0
num_v = 0

for filename in os.listdir(v_input_dir):
    if filename.endswith(".jpg") :  
        input_path = os.path.join(v_input_dir, filename)
        print(f"Processing: {input_path}")

        img = Image.open(input_path)
        img = np.array(img)
        img = preprocess_frame(img)
        img = np.expand_dims(img, axis=0)


        predicted_labels_probs = model.predict(img)[0]
        predicted_label_index = np.argmax(predicted_labels_probs)
        predicted_class_name = CLASSES_LIST[predicted_label_index]
        num_v = num_v + 1
        if predicted_class_name == "Violence":
            num_v_correct = num_v_correct + 1
            true_positives = true_positives + 1
        else:
            false_negatives = false_negatives + 1


testing_acc = (num_nv_correct + num_v_correct)/(num_nv + num_v)
precision = true_positives / (true_positives + false_positives + 1e-7)
recall = true_positives / (true_positives + false_negatives + 1e-7)

print("testing accuracy is: " + str(testing_acc))
print("precision: " + str(precision))
print("recall: " + str(recall))