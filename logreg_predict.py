import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOUSES_COL = "Hogwarts House"
SELECTED_FEATURES = ["Astronomy","Herbology","Defense Against the Dark Arts","Ancient Runes","Charms"]
T0_LABEL = "t0"

def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

def format_list(W):
    W_tmp = []
    for each in W :
        W_tmp.append(float(each))
    return W_tmp

def predict(X, W):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    A = []
    len_feature = len(SELECTED_FEATURES)
    mapping = [float] * len_feature
    mapping = np.reshape(mapping, (1,len_feature))
    for j in range(len(X[1])) :
        for i in range (0, len_feature) :
            mapping[0][i] = X[i][j]
        for house in range(len(houses)) :
            tmp = format_list(W[house])
            tmp = pd.DataFrame(tmp)
            tmp = tmp.values.T
            Z = float(np.dot(tmp, np.matrix(mapping).T))
            A.append(sigmoid(Z))
    return A

def clean_dataframe(df, std, mean):
    selected_features = [HOUSES_COL] + [T0_LABEL] + SELECTED_FEATURES
    X_test = normalize(df.loc[:, selected_features[2:]], std, mean)
    X_test = X_test.fillna(method='ffill') # Verif
    X_test = X_test.values.T
    return X_test

def normalize(X, std, mean):
    return (X - mean) / std

def parse_weight(W):
    std = []
    mean = []
    try :
        weight = pd.read_csv("./datasets/weights.csv")
    except :
        print("Can't open file")
        return 0
    for i in range (4):
        W.append(list(weight.iloc[i]))

    std = list(weight.iloc[4])
    mean = list(weight.iloc[5])
    return W, std, mean

def choixpeau(predicted):
    count = 0
    house = []
    max = [0.0,0]
    i = 1
    for each in predicted :
        if each > max[0] :
            max[0] = float(each)
            max[1] = i
        if i == 4:
            i = 0
            house.append(max[1])
            max = [0.0,0]
        i += 1
        count += 1
    return (house)

def print_choix(prediction, filename='./houses.csv') :
    f = open(filename, 'w+')
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    i = 0
    f.write("Index,Hogwarts House\n")
    for each in prediction :
        f.writelines([f'{i}',',',f'{houses[each - 1]}','\n'])
        i += 1

def main(filename):
    W = []
    prediction = []
    try :
        X_test = pd.read_csv(filename)
    except :
        print("Can't open file")
        return 0
    W, std, mean = parse_weight(W)
    X_test = clean_dataframe(X_test, std, mean)
    prediction = predict(X_test, W)
    prediction = choixpeau(prediction)
    print_choix(prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Input dataset")
    args = parser.parse_args()

    main(args.dataset)