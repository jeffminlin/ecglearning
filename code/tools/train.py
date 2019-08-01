import sys

import pandas as pd
import numpy as np
import tensorflow as tf

from . import models as mod


def preprocess(trainfile, testfile, fft=False):

    filenames = {"train": trainfile, "test": testfile}
    datasets = {}
    dataset_arrays = {}
    labels = {}
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

        # Isolate data and convert to numpy array
        dataset_arrays[key] = datasets[key].drop("Classes", axis=1)
        dataset_arrays[key] = dataset_arrays[key].to_numpy()

    if fft:
        dataset_arrays = dataset_fft(dataset_arrays)

    return dataset_arrays, labels, datasets


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


def train_dense(datasets, labels, hiddenlayers, config):

    inputsize = datasets["train"].shape[1]
    ncategories = labels["train"].shape[1]
    model = mod.create_dense(inputsize, hiddenlayers, ncategories)
    model.summary()

    print()
    print("Training")
    model.compile(
        optimizer="Nadam",
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.categorical_accuracy],
    )
    model.fit(
        datasets["train"],
        labels["train"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=config["val_split"],
    )
    print()
    print("Testing")
    model.evaluate(datasets["test"], labels["test"])
