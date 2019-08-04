import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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
    earlystop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        restore_best_weights=True,
        min_delta=1.0e-4,
    )
    tensorboard_callback = callbacks.TensorBoard(log_dir=config["logdir"])
    model.compile(
        optimizer=config["optimizer"],
        loss=config["loss"],
        metrics=[tf.keras.metrics.categorical_accuracy],
    )
    history = model.fit(
        dataset_arrays["train"],
        labels["train"],
        verbose=config["verbose"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=[earlystop, tensorboard_callback],
        validation_split=config["val_split"],
    )
    print(
        "Train acc:",
        model.evaluate(dataset_arrays["train"], labels["train"], verbose=0)[1],
    )
    print(
        "Test acc:",
        model.evaluate(dataset_arrays["test"], labels["test"], verbose=0)[1],
    )

    return history


def plot_fit_history(history):

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(history.history["loss"], ".-", label="loss")
    try:
        plt.plot(history.history["val_loss"], ".-", label="val_loss")
    except KeyError:
        print("No validation loss")
    plt.legend()
    plt.show()


def plot_cm(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function from the scikit-learn website prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print("Confusion matrix, without normalization")

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.show()
