import pandas as pd
import matplotlib.pyplot as plt


def plot_ecg(filename, num):

    df = pd.read_csv(filename, header=None, nrows=num)
    df.T.plot(subplots=True, title="The first " + str(num) + " heartbeats")
    plt.show()


def main():

    plot_ecg("../data/mitbih_test.csv", 4)


if __name__ == "__main__":

    main()
