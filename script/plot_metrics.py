import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def PlotMetric(csvFile: str):
    if (csvFile.find("/", 0, 1) == -1):
        csvFile = "/" + csvFile

    currentDir = os.getcwd()
    fullPathCsv = f"{currentDir}{csvFile}"

    df = pd.read_csv(fullPathCsv)

    epoch = df["epoch"]
    batch = df["batch"]
    accuracy = df["accuracy"]

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.plot(epoch, accuracy, linewidth=1.5)
    ax.set_xlabel('batch (qt)')
    ax.set_ylabel('accuracy (%)')
    ax.set_title("Batch x Accuracy")
    ax.grid()
    # ax.legend()

    pathSaveFig = f"{currentDir}/plot/accuracy_batch_4.png"

    fig.savefig(pathSaveFig, dpi=200)

PlotMetric("plot/net_metric.csv")
