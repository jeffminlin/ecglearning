"""Plot some random heartbeats from a chosen dataset."""
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_processed_data(df, title, xlabel, ylabel, process, num):
    """Make a figure with subplots after processing the data."""

    num = len(df.index)
    fig, axes = plt.subplots(num, 1, sharex=True, figsize=(8, 2.5*num))
    fig.suptitle(title)
    for idx in range(num):
        axes[idx].title.set_text("Sample number " + str(df.index[idx]))
        axes[idx].set_ylabel(ylabel)
        axes[idx].plot(
            *process(df.iloc[idx]), label="Class: " + str(int(df.iloc[idx, -1]))
        )
        axes[idx].legend()
    # Shared xlabel
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    return fig


def plot_ecg(filename, samplerate, num):
    """Assumes that the last column is the class."""

    df = pd.read_csv(filename, header=None)
    df.rename(columns={df.columns[-1]: "Classes"}, inplace=True)

    data = df.groupby("Classes", group_keys=False).apply(
        pd.DataFrame.sample, n=num, axis=0
    )
    fig = plot_processed_data(
        data,
        "Random samples of each type of heartbeat in " + filename,
        "seconds",
        "",
        lambda row: (
            np.arange(len(np.trim_zeros(row[:-1]))) / samplerate,
            np.trim_zeros(row[:-1]),
        ),
        len(data.index)
    )
    fig_freq = plot_processed_data(
        data,
        "Magnitude responses of each heartbeat",
        "Hz",
        "",
        lambda row: (
            np.fft.rfftfreq(len(np.trim_zeros(row[:-1])), d=(1.0 / samplerate)),
            np.abs(np.fft.rfft(np.trim_zeros(row[:-1]))),
        ),
        len(data.index)
    )

    plt.show()


def plot_fit_history(history):

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(history.history["loss"], ".-", label="loss")
    if history.history.get("val_loss"):
        plt.plot(history.history["val_loss"], ".-", label="val_loss")
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
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred).astype(int)]
    if normalize:
        print("Confusion matrix, without normalization")
        print(cm)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:


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
