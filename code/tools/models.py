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


def create_conv1d(inputsize, layerlist, ncategories, config):
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
            outputs = layers.MaxPooling1D(pool_size=layer[1], strides=2)(outputs)
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
    outputs = layers.Dense(
        ncategories, activation="softmax", kernel_regularizer=config.get("regularizer")
    )(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="conv1d")


# WaveNet components
def filter_gate_multiply(nfilters, dilation_rate, output):

    filter_out = layers.Conv1D(
        filters=nfilters,
        kernel_size=2,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="tanh",
        kernel_initializer="glorot_normal",
    )(output)
    gate = layers.Conv1D(
        filters=nfilters,
        kernel_size=2,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="sigmoid",
        kernel_initializer="glorot_normal",
    )(output)

    return layers.Multiply()([filter_out, gate])


def residual_block(nfilters, dilation_rate):

    return [
        ("startskip",),
        (lambda output: filter_gate_multiply(nfilters, dilation_rate, output),),
        (
            "conv",
            {
                "filters": nfilters,
                "width": 1,
                "padding": "causal",
                "dilation": 1,
                "activation": "linear",
            },
        ),
        ("startskip",),
        ("endskip", -2),
    ]


def add_res_blocks(nblocks, nfilters, dilation_limit, layerlist):

    dilation_rate = 1
    for block_idx in range(nblocks):
        if dilation_rate > dilation_limit:
            dilation_rate = 1
        layerlist.extend(residual_block(nfilters, dilation_rate))
        dilation_rate *= 2
    for skip_idx in range(nblocks):
        layerlist.append(("endskip",))


def create_lstm(LSTM_list, ncategories, config):

    inputs = tf.keras.Input(shape=(None, 1))
    outputs = inputs
    for layer in LSTM_list:
        outputs = layers.LSTM(layer[0])(outputs)
    outputs = layers.Dense(
        ncategories, activation="softmax", kernel_regularizer=config.get("regularizer")
    )(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm")
