import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

def format_list(W):
    W_tmp = []
    for each in W :
        each = each.replace("[", "")
        each = each.replace("]", "")
        print(each)
        W_tmp.append(float(each))
    return W_tmp

def predict(X, W):
    features = ["Astronomy","Herbology","Defense Against the Dark Arts","Divination","Ancient Runes","Charms"]
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    A = []
    len_feature = len(features)
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

def clean_dataframe(X_test):
    droped = ["Index","Arithmancy","Muggle Studies","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Flying","Hogwarts House"]
    X_test = X_test.select_dtypes("number")
    
    for each in droped :
        X_test = X_test.drop(each, axis=1)

    X_test = X_test.dropna()
    X_test = X_test.values.T

    return X_test

def normalize(X):
    return (X - X.mean()) / X.std()

def parse_weight(W):
    try :
        weight = pd.read_csv("./datasets/weights.csv")
    except :
        print("Can't open file")
        return 0
    print(list(weight.iloc[0]))
    for i in range (4):
        W.append(list(weight.iloc[i]))
    return W

# def visualizastion():
    # for each in houses :
        # plt.plot(np.arange(iter), cost_list)
        # plt.title(each)
        # plt.ylabel("Cost")
        # plt.xlabel("Itération")
        # plt.show()

def choixpeau(predicted):
    count = 0
    house = []
    max = [0.0,0]
    print(len(predicted))
    i = 1
    for each in predicted :
        print(i, " : ", each)
        if each > max[0] :
            max[0] = float(each)
            max[1] = i
        if i == 4:
            i = 0
            house.append(max[1])
            max = [0.0,0]
        i += 1
        count += 1
    # house.append(max[1])
    return (house)

def main(filename):
    W = []
    prediction = []
    # features = ["Index","Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    try :
        X_test = pd.read_csv(filename)
    except :
        print("Can't open file")
        return 0
    W = parse_weight(W)
    X_test = clean_dataframe(X_test)
    X_test = normalize(X_test)
    prediction = predict(X_test, W)
    i = 0
    # for each in prediction :
    #     if i % 4 == 0 :
    #         print(i / 4, "\n")
    #     print(each, "\n")
    #     i += 1
    print(choixpeau(prediction))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Input dataset")
    args = parser.parse_args()

    main(args.dataset)