import sys

import pandas as pd
import numpy as np
import tensorflow as tf

import tools.models


def preprocess(trainfile, testfile):

    filenames = {"train": trainfile, "test": testfile}
    datasets = {}
    labels = {}
    for key, name in filenames.items():
        datasets[key] = pd.read_csv(name, header=None)
        datasets[key].rename(
            columns={datasets[key].columns[-1]: "Classes"}, inplace=True
        )
        datasets[key]["Classes"].astype("int")
        print(key.capitalize() + " set")
        print("Count of each class")
        print(datasets[key].groupby("Classes").size())
        print()
        labels[key] = tf.keras.utils.to_categorical(
            datasets[key]["Classes"],
            num_classes=len(datasets[key].groupby("Classes").size()),
        )

    return datasets, labels


def train_dense(datasets, labels, hiddenlayers, ncategories):

    inputsize = len(datasets["train"].columns) - 1
    hiddenlayers = [(32, "relu"), (32, "relu")]
    model = tools.models.create_dense(inputsize, hiddenlayers, ncategories)
    model.summary()


def main():

    pass


if __name__ == "__main__":

    main()
