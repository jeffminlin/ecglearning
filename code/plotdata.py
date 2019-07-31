"""Plot some random heartbeats from a chosen dataset."""
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_processed_data(df, title, xlabel, ylabel, process):
    """Make a figure with subplots after processing the data."""

    num = len(df.index)
    fig, axes = plt.subplots(num, 1, sharex=True)
    fig.suptitle(title)
    for idx in range(num):
        axes[idx].title.set_text("Sample number " + str(df.index[idx]))
        axes[idx].set_ylabel(ylabel)
        axes[idx].plot(
            process(df.iloc[idx]), label="Class: " + str(int(df.iloc[idx, -1]))
        )
        axes[idx].legend()
    # Shared xlabel
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    return fig


def plot_ecg(filename, num):
    """Assumes that the last column is the class."""

    df = pd.read_csv(filename, header=None)
    df.rename(columns={df.columns[-1]: "Classes"}, inplace=True)
    print("Frequency of each class")
    print(df.groupby("Classes").size())
    data = df.sample(n=num, axis=0)
    fig = plot_processed_data(
        data,
        "Random " + str(num) + " heartbeats in " + filename,
        "",
        "",
        lambda row: np.trim_zeros(row[:-1]),
    )
    fig_freq = plot_processed_data(
        data,
        "Magnitude response of random " + str(num) + " heartbeats",
        "Frequency",
        "",
        lambda row: np.abs(np.fft.rfft(np.trim_zeros(row[:-1]))),
    )

    plt.show()


def main():
    """Usage:
    python path-to-this-file path-to-data number-of-samples
    e.g. from the main project folder, you might run
        python ./code/plotdata.py ./data/mitbih_train.csv 3
    """

    plot_ecg(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":

    main()
