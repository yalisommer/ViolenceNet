"""
Homework 5 - CNNs
CSCI1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

import hyperparameters as hp

from keras import config
# Call before any model definitions
config.set_dtype_policy("float32")

class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()
        # should use stochastic gradient descent

        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = hp.learning_rate, # may need to change value for SGD
            momentum = hp.momentum
              # might want clipnorm, clipvalue
        )
    
        self.architecture = [
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
        ]

        #       Don't change the line below. This line creates an instance
        #       of a Sequential model using the layers you defined above. 
        #       A sequential model, when called, calls its own layers in 
        #       order to produce its output! 
        self.your_model = tf.keras.Sequential(self.architecture, name="your_model")

    def call(self, x):
        """ Passes input image through the network. """

        x = self.your_model(x)

        #       Note: If we hadn't defined the Sequential instance, the below 
        #       lines would achieve the same output!
        # for layer in self.architecture:
        #     x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TASK 1
        # TODO: Select a loss function for your network 
        #       (see the documentation for tf.keras.losses)
       
       # do labels count as true? uncelar !!
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions) # unclear if labels is y_true, but prob, also unsure if i shold use car_crossE


# class VGGModel(tf.keras.Model):
#     def __init__(self):
#         super(VGGModel, self).__init__()

#         # TASK 3
#         # TODO: Select an optimizer for your network (see the documentation
#         #       for tf.keras.optimizers)
#         self.optimizer = tf.keras.optimizers.SGD(
#             learning_rate = hp.learning_rate,
#             momentum = hp.momentum
#         )

#         # Don't change the below:

#         self.vgg16 = [
#             # Block 1
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv1"),
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv2"),
#             MaxPool2D(2, name="block1_pool"),
#             # Block 2
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv1"),
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv2"),
#             MaxPool2D(2, name="block2_pool"),
#             # Block 3
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv1"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv2"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv3"),
#             MaxPool2D(2, name="block3_pool"),
#             # Block 4
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv3"),
#             MaxPool2D(2, name="block4_pool"),
#             # Block 5
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv3"),
#             MaxPool2D(2, name="block5_pool")
#         ]

#         # TASK 3
#         # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
#         #       pretrained VGG16 weights into place so that only the classificaiton
#         #       head is trained.

#         for layer in self.vgg16:
#             layer.trainable = False


#         self.head = [
#               # flatten?
#             Flatten(),
#             Dense(128, activation='relu'),
#             Dense(128, activation='relu'),
#             Dense(15, activation='softmax')
#         ]

#         # Don't change the below:
#         self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
#         self.head = tf.keras.Sequential(self.head, name="vgg_head")

#     def call(self, x):
#         """ Passes the image through the network. """

#         x = self.vgg16(x)
#         x = self.head(x)

#         return x

#     @staticmethod
#     def loss_fn(labels, predictions):
#         """ Loss function for model. """

#         return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
