import sys

import pandas as pd
import numpy as np
import tensorflow as tf

import models


def preprocess(trainfile, testfile, fft=False):

    filenames = {"train": trainfile, "test": testfile}
    datasets = {}
    dataset_arrays = {}
    labels = {}
    for key, name in filenames.items():
        # Read in dataset
        datasets[key] = pd.read_csv(name, header=None)
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
        print(key.capitalize() + " set")
        print("Count of each class")
        print(datasets[key].groupby("Classes").size())
        print()

def dataset_fft(dataset_arrays):

    datasets_fft = {}
    for key in dataset_arrays:
        datasets_fft[key] = np.abs(np.fft.rfft(dataset_arrays[key]))

    return datasets_fft


def train_dense(datasets, labels, hiddenlayers, config):

    inputsize = datasets["train"].shape[1]
    ncategories = labels["train"].shape[1]
    model = models.create_dense(inputsize, hiddenlayers, ncategories)
    model.summary()

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
    model.evaluate(datasets["test"], labels["test"])


def main():

    # Read in data
    files = (sys.argv[1], sys.argv[2])
    inputs, labels, df = preprocess(*files, fft=True)
    class_count(df)

    # Try least-squares
    print("Trying least-squares")
    lstsq_soln = np.linalg.lstsq(inputs["train"], labels["train"], rcond=None)
    print("Rank of training dataset:", lstsq_soln[2])
    print()

    print("Checking predictions")
    coeffs = lstsq_soln[0]
    predict = {}
    accuracy = {}
    for key in inputs:
        predict[key] = np.argmax(np.dot(inputs[key], coeffs), axis=1)
        num_correct = np.sum(
            labels[key][range(labels[key].shape[0]), predict[key]] == 1
        )
        accuracy[key] = num_correct / labels[key].shape[0]
    print("Training accuracy:", accuracy["train"])
    print("Test accuracy:", accuracy["test"])
    print()

    # Try a bland dense network
    # Would probably be better to avoid overfitting in some way
    print("Trying a bland dense network")
    config = {"batch_size": 30, "val_split": 0.2, "epochs": 20}
    hiddenlayers = [(10, "relu"), (10, "relu")]
    train_dense(inputs, labels, hiddenlayers, config)


if __name__ == "__main__":

    main()
