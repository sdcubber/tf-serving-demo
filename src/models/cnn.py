import numpy as np
import tensorflow as tf

from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Input, Flatten
from keras.models import Model

class CNN_classifier(object):

    def __init__(self, im_size,  n_labels):
        """
        CNN for multi-label image classification with binary relevance
        """

        self.im_size = im_size
        self.n_labels = n_labels
        self.dropout_rate = 0.15
        self.n_neurons = 128  # Number of neurons in dense layers
        # build model on init
        self.build()

    def build(self):
        # Define input
        self.x = tf.keras.layers.Input(shape=(self.im_size, self.im_size, 3))

        # Convolutional layers
        conv_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(self.x)
        conv_1 = tf.keras.layers.MaxPooling2D(padding='same')(conv_1)
        conv_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                        padding='same', activation='relu')(conv_1)
        conv_2 = tf.keras.layers.MaxPooling2D(padding='same')(conv_2)

        # Flatten
        conv_flat = tf.keras.layers.Flatten()(conv_2)
        # Fully connected layers
        fc_1 = tf.keras.layers.Dense(self.n_neurons, activation='relu')(conv_flat)
        fc_1 = tf.keras.layers.Dropout(self.dropout_rate)(fc_1)
        fc_2 = tf.keras.layers.Dense(self.n_neurons, activation='relu')(fc_1)
        self.fc_2 = tf.keras.layers.Dropout(self.dropout_rate)(fc_2)

        # Output layers: n_classes output nodes for binary relevance
        self.y = tf.keras.layers.Dense(self.n_labels, activation='sigmoid')(self.fc_2)

        self.model = tf.keras.models.Model(inputs=self.x, outputs=self.y)


class BlogpostCNN(object):

    def __init__(self, im_size, w_labels, g_labels):
        """
        CNN for multi-class and multi-label classification
        """

        self.im_size = im_size
        self.w_labels = w_labels
        self.g_labels = g_labels
        self.dropout_rate = 0.15
        self.n_neurons = 128  # Number of neurons in dense layers
        # build model on init
        self.build()

    def build(self):
        # Define input
        self.x = Input(shape=(self.im_size, self.im_size, 3))

        # Convolutional layers
        conv_1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(self.x)
        conv_1 = MaxPooling2D(padding='same')(conv_1)
        conv_2 = Conv2D(32, kernel_size=(3, 3),
                        padding='same', activation='relu')(conv_1)
        conv_2 = MaxPooling2D(padding='same')(conv_2)

        # Flatten
        conv_flat = Flatten()(conv_2)

        # Fully connected layers
        fc_1 = Dense(self.n_neurons, activation='relu')(conv_flat)
        fc_1 = Dropout(self.dropout_rate)(fc_1)
        fc_2 = Dense(self.n_neurons, activation='relu')(fc_1)
        self.fc_2 = Dropout(self.dropout_rate)(fc_2)

        # Output layers: n_classes output nodes for binary relevance
        self.weather = Dense(self.w_labels, activation='softmax')(self.fc_2)
        self.ground = Dense(self.g_labels, activation='sigmoid')(self.fc_2)

        self.model = Model(inputs=self.x, outputs=[self.weather, self.ground])