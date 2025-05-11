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

        #       This line creates an instance
        #       of a Sequential model using the layers you defined above. 
        #       A sequential model, when called, calls its own layers in 
        #       order to produce its output! 
        self.your_model = tf.keras.Sequential(self.architecture, name="your_model")

    def call(self, x):
        """ Passes input image through the network. """

        x = self.your_model(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions) 