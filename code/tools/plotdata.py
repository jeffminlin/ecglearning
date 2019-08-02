"""Plot some random heartbeats from a chosen dataset."""
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
