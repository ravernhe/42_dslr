import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Histogram:
    def __init__(self, df, name, legend, xlabel, ylabel) -> None:
        self.df = df
        self.name = name
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel

    def show_histogram(self):
        
        x = self.df[self.name][:326].to_numpy()
        y = self.df[self.name][326:855].to_numpy()
        z = self.df[self.name][855:1298].to_numpy()
        a = self.df[self.name][1298:].to_numpy()

        plt.hist(x, histtype="stepfilled", align="mid", color="red", alpha=0.5)
        plt.hist(y, histtype="stepfilled", align="mid", color="yellow", alpha=0.5)
        plt.hist(z, histtype="stepfilled", align="mid", color="blue", alpha=0.5)
        plt.hist(a, histtype="stepfilled", align="mid", color="green", alpha=0.5)

        plt.title(self.name)
        plt.legend(self.legend, loc='upper right', frameon=False)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()
