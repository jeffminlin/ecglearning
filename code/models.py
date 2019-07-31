import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


class DenseModel(tf.keras.Model):
    def __init__(self, hiddenlayersizes, ncategories):
        super(DenseModel, self).__init__(name="dense_model")
        self.inputlayer = layers.Dense(None, activation="relu")
        self.hiddenlayers = []
        for size in hiddenlayersizes[:-1]:
            self.hiddenlayers.append(layers.Dense(size, activation="relu"))
        self.hiddenlayers.append(layers.Dense(hiddenlayersizes[-1], activation="tanh"))
        self.outputlayer = layers.Dense(ncategories)

    def call(self, inputs):
        x = self.inputlayer(inputs)
        for layer in self.hiddenlayers:
            x = layer(x)
        return self.outputlayer(x)
