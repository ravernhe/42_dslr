import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Scatter_plot:
    def __init__(self, df, legend, xlabel, ylabel) -> None:
        self.df = df
        self.legend =legend
        self.xlabel = xlabel
        self.ylabel = ylabel

    def show(self):
        plt.scatter(self.df[self.xlabel][:326], self.df[self.ylabel][:326], color='red', alpha=0.5)
        plt.scatter(self.df[self.xlabel][326:855], self.df[self.ylabel][326:855], color='yellow', alpha=0.5)
        plt.scatter(self.df[self.xlabel][855:1298], self.df[self.ylabel][855:1298], color='blue', alpha=0.5)
        plt.scatter(self.df[self.xlabel][1298:], self.df[self.ylabel][1298:], color='green', alpha=0.5)

        plt.legend(self.legend, loc='upper right', frameon=False)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()