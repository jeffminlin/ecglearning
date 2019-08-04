import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


def create_dense(inputsize, hiddenlayers, ncategories):
    """Make a dense fully-connected network.
    
    To include a hidden layer, put a tuple (size, activation) in hiddenlayers.
    """

    inputs = tf.keras.Input(shape=(inputsize))
    outputs = layers.Dense(hiddenlayers[0][0], activation=hiddenlayers[0][1])(inputs)
    for layer in hiddenlayers[1:]:
        outputs = layers.Dense(layer[0], activation=layer[1])(outputs)
    outputs = layers.Dense(ncategories, activation="softmax")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dense_model")


def create_conv1d(inputsize, layerlist, ncategories):
    """Make a CNN with convolutional layers, max pooling, and fully connected layers.

    The convolutional layers will have 'glorot_normal' initialization. To add one,
    include the tuple ("conv", conv_dict) in layerlist, where conv_dict is a dictionary
    minimally including the keys "filters" (number of filters) and
    "width" (kernel size).

    The max pooling layers will have 'valid' padding. For a max pooling layer,
    include the tuple ("maxpool", pool_size) in layerlist.

    To include a fully-connected layer, include the tuple ("fc", size, activation).

    To include identity elements, use ("startskip") and ("endskip") to add and remove
    layers to a list. The default behavior is to use a stack. You can specify
    ("endskip", 4) for finer-grained control over the element that is re-added. For
    example, using only ("endskip", 0) would use a queue instead of a stack.

    You can add arbitrary layers by including (layertobeadded) in layerlist.
    """

    inputs = tf.keras.Input(shape=(inputsize, 1))
    outputs = inputs
    checkpoints = []
    for layer in layerlist:
        if layer[0] == "conv":
            # Defaults
            if not layer[1].get("padding"):
                layer[1]["padding"] = "same"
            if not layer[1].get("dilation"):
                layer[1]["dilation"] = 1
            if not layer[1].get("activation"):
                layer[1]["activation"] = "relu"
            # Construction
            outputs = layers.Conv1D(
                filters=layer[1]["filters"],
                kernel_size=layer[1]["width"],
                padding=layer[1]["padding"],
                dilation_rate=layer[1]["dilation"],
                activation=layer[1]["activation"],
                kernel_initializer="glorot_normal",
            )(outputs)
        elif layer[0] == "maxpool":
            outputs = layers.MaxPooling1D(pool_size=layer[1])(outputs)
        elif layer[0] == "fc":
            outputs = layers.Dense(layer[1], activation=layer[2])(outputs)
        elif layer[0] == "startskip":
            checkpoints.append(outputs)
        elif layer[0] == "endskip":
            if len(layer) < 2:
                skipped = checkpoints.pop()
            else:
                skipped = checkpoints[layer[1]]
                del checkpoints[layer[1]]
            outputs = layers.Add()([outputs, skipped])
        else:
            outputs = layer[0](outputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(ncategories, activation="softmax")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="conv1d")

