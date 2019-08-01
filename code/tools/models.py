import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


def create_dense(inputsize, hiddenlayers, ncategories):

    inputs = tf.keras.Input(shape=(inputsize))
    outputs = layers.Dense(hiddenlayers[0][0], activation=hiddenlayers[0][1])(inputs)
    for size, activation in hiddenlayers[1:]:
        output = layers.Dense(size, activation=activation)(outputs)
    outputs = layers.Dense(ncategories, activation="tanh")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dense_model")
