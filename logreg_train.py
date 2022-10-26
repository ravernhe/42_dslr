import argparse
from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from describe import Describe

SELECTED_FEATURES = ["Defense Against the Dark Arts","Divination","Charms", "History of Magic", "Ancient Runes"]

class LogReg():
    def __init__(self, X, iter, eta) -> None:
        self.W = []
        self.iter = iter
        self.eta = eta
        self.X = X
        self.y = None

    def one_vs_all(self) :
        for house in self.X.index.unique():
            self.y = self.X.index.map(lambda x: 0 if x != house else 1)
            self.binary_classifier()
        print(self.W)

    def binary_classifier(self):
        W = np.zeros(len(self.X.columns))

        for _ in range(self.iter):
            Z = np.dot(self.X, W)
            A = self.sigmoid(Z)
            

            dW = np.dot(self.X.T, (A - self.y)) / self.y.size
            
            W -= self.eta * dW.T

        self.W.append(W)

    def sigmoid(self, Z):
        return (1.0 / (1.0 + np.exp(-Z)))


def standardize(X, mean, std):
    return (X - mean) / std

def logreg_train(iter, eta):
    describe = Describe("./datasets/dataset_train.csv")
    describe.agregate()
    agregated_df = describe.aggregated_df[SELECTED_FEATURES]
    df = describe.df[SELECTED_FEATURES]
    standardize_df = pd.DataFrame()

    for col in df:
        standardize_df[col] = standardize(df[col], agregated_df[col]["mean"], agregated_df["Charms"]["std"])

    standardize_df = standardize_df.fillna(0)
    logreg = LogReg(standardize_df, iter, eta)
    logreg.one_vs_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, help="Nb of iteration", default=10000)
    parser.add_argument("-e", "--eta", type=float, help="Learning rate", default=0.0015)
    args = parser.parse_args()
    logreg_train(args.iter, args.eta)