import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from keras_tqdm import TQDMNotebookCallback


def preprocess(trainfile, testfile, fft=False):

    filenames = {"train": trainfile, "test": testfile}
    datasets = {}
    dataset_arrays = {}
    labels = {}
    sparse_labels = {}
    for key, name in filenames.items():
        # Read in dataset
        datasets[key] = pd.read_csv(name, header=None)
        # Shuffle in-place
        datasets[key] = datasets[key].sample(frac=1)
        datasets[key].rename(
            columns={datasets[key].columns[-1]: "Classes"}, inplace=True
        )
        datasets[key]["Classes"].astype("int")

        # Make labels
        labels[key] = tf.keras.utils.to_categorical(
            datasets[key]["Classes"],
            num_classes=len(datasets[key].groupby("Classes").size()),
        )
        sparse_labels[key] = datasets[key]["Classes"].to_numpy()

        # Isolate data and convert to numpy array
        dataset_arrays[key] = datasets[key].drop("Classes", axis=1)
        dataset_arrays[key] = dataset_arrays[key].to_numpy()

    if fft:
        dataset_arrays = dataset_fft(dataset_arrays)

    return dataset_arrays, labels, sparse_labels, datasets


def class_count(datasets):

    for key in datasets:
        print()
        print(key.capitalize() + " set")
        print("Count of each class")
        print(datasets[key].groupby("Classes").size())


def dataset_fft(dataset_arrays):

    datasets_fft = {}
    for key in dataset_arrays:
        datasets_fft[key] = np.abs(np.fft.rfft(dataset_arrays[key]))

    return datasets_fft


def train_print(model, dataset_arrays, labels, config):

    model.summary()

    print()
    print("Training")
    history = train(model, dataset_arrays, labels, config)
    print(
        "Train acc:",
        model.evaluate(dataset_arrays["train"], labels["train"], verbose=0)[1],
    )
    print(
        "Test acc:",
        model.evaluate(dataset_arrays["test"], labels["test"], verbose=0)[1],
    )

    return history


def train(model, dataset_arrays, labels, config):
    """All the training without any of the extra printing."""

    earlystop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        restore_best_weights=True,
        min_delta=1.0e-4,
    )
    callback_list = [earlystop]
    if config.get("progbar"):
        tqdmcb = TQDMNotebookCallback()
        tqdmcb.on_train_batch_begin = tqdmcb.on_batch_begin
        tqdmcb.on_train_batch_end = tqdmcb.on_batch_end
        tqdmcb.on_test_batch_begin = tqdmcb.on_batch_begin
        tqdmcb.on_test_batch_end = tqdmcb.on_batch_end
        setattr(tqdmcb, "on_test_begin", lambda x: None)
        setattr(tqdmcb, "on_test_end", lambda x: None)
        callback_list.append(tqdmcb)
    if config.get("logdir"):
        tensorboard_callback = callbacks.TensorBoard(log_dir=config["logdir"])
        callback_list.append(tensorboard_callback)
    model.compile(
        optimizer=config["optimizer"],
        loss=config["loss"],
        metrics=[tf.keras.metrics.categorical_accuracy],
    )
    history = model.fit(
        dataset_arrays["train"],
        labels["train"],
        class_weight=config.get("class_weights"),
        verbose=config["verbose"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=callback_list,
        validation_split=config["val_split"],
    )

    return history
