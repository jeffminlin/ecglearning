import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


def create_dense(inputsize, hiddenlayers, ncategories):

    inputs = tf.keras.Input(shape=(inputsize))
    outputs = layers.Dense(hiddenlayers[0][0], activation=hiddenlayers[0][1])(inputs)
    for layer in hiddenlayers[1:]:
        outputs = layers.Dense(layer[0], activation=layer[1])(outputs)
    outputs = layers.Dense(ncategories, activation="sigmoid")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dense_model")
