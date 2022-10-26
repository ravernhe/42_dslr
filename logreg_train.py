import argparse
from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from describe import Describe
import csv

SELECTED_FEATURES = ["Defense Against the Dark Arts","Divination","Charms", "History of Magic", "Ancient Runes"]

class LogReg():
    def __init__(self, X, iter, eta) -> None:
        self.X = X
        self.iter = iter
        self.eta = eta

        self.y = None
        self.cost = {}
        self.W = {}

    def save_as_csv(self):
        with open('datasets/weights.csv', 'w') as f:
            w = csv.writer(f,)
            w.writerow(self.W.keys())
            for c in range(len(SELECTED_FEATURES)):
                row = [self.W[house][c] for house in self.W.keys()]
                w.writerow(row)

    def one_vs_all(self) :
        for house in self.X.index.unique():
            self.y = self.X.index.map(lambda x: 0 if x != house else 1)
            self.cost[house], self.W[house] = self.binary_classifier()
        self.save_as_csv()

    def binary_classifier(self):
        m, n = self.X.shape
        W = np.zeros(n)
        cost_list = []
        
        for _ in range(self.iter):
            Z = np.dot(self.X, W)
            A = self.sigmoid(Z)
            
            dW = np.dot(self.X.T, (A - self.y)) / self.y.size
            W -= self.eta * dW.T

            cost_list.append((1 / m) * (np.dot(-self.y.T, np.log(A)) - np.dot((1 - self.y).T, np.log(1 - A))))

        return cost_list, W.tolist()

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