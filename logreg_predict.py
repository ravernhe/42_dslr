import argparse
import csv
import numpy as np
import pandas as pd
from describe import Describe

SELECTED_FEATURES = ["Defense Against the Dark Arts","Divination","Charms", "History of Magic", "Ancient Runes"]

class LogRegPredict():
    def __init__(self, X, W) -> None:
        self.X = X
        self.W = W
        self.predict_score = []

    def sigmoid(self, Z):
        return (1.0 / (1.0 + np.exp(-Z)))

    def predict(self):
        for i, row in self.X.iterrows():
            score = {}
            for house in self.W.columns:
                Z = np.dot(row, self.W[house])
                A = self.sigmoid(Z)
                score[house] = A
            self.predict_score.append(score)
    
    def choixpeau(self):
        house = []
        with open('houses.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(["Index", "Hogwarts House"])
            for i, row in enumerate(self.predict_score):
                w.writerow([i, max(row, key=row.get)])



def print_choix(prediction, filename='./houses.csv') :
    f = open(filename, 'w+')
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    i = 0
    f.write("Index,Hogwarts House\n")
    for each in prediction :
        f.writelines([f'{i}',',',f'{houses[each - 1]}','\n'])
        i += 1

def standardize(X, mean, std):
    return (X - mean) / std

def get_weight():
    try :
        return pd.read_csv("./datasets/weights.csv")
    except :
        raise Exception("File not found or error while opening the file")

def log_reg_predict():
    describe = Describe("./datasets/dataset_test.csv")
    describe.agregate()
    agregated_df = describe.aggregated_df[SELECTED_FEATURES]
    df = describe.df[SELECTED_FEATURES]
    standardize_df = pd.DataFrame()

    for col in df:
        standardize_df[col] = standardize(df[col], agregated_df[col]["mean"], agregated_df[col]["std"])
    standardize_df = standardize_df.fillna(0)

    W = get_weight()

    prediction = LogRegPredict(standardize_df, W)
    prediction.predict()
    prediction.choixpeau()

if __name__ == "__main__":
    log_reg_predict()