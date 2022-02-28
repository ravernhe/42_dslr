import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOUSES_COL = "Hogwarts House"
SELECTED_FEATURES = ["Astronomy","Herbology","Defense Against the Dark Arts","Ancient Runes","Charms"]
T0_LABEL = "t0"

class Log_reg():
    def __init__(self, iter=10000, eta=0.0015) -> None:
        # self.df = df
        self.houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
        self.W = []
        self.cost = []
        self.iter = iter
        self.eta = eta

    def one_vs_all(self, X_train, Y_train) :
        for each in self.houses :
            f = lambda x: 0 if x != each else 1
            Y = Y_train.map(f)
            Y = Y.values
            self.binary_classifier(X_train, Y, self.eta, self.iter)

    def binary_classifier(self, X_train, Y_train, eta, iter):
        m, n = X_train.shape
        W = np.zeros(n)
        # B = 0

        cost_list = []
        for i in range(self.iter):
            Z = np.dot(X_train, W)
            A = self.sigmoid(Z)
            
            # cost function
            # cost = -(1.0 / m) * np.sum(Y_train * np.log(A) + (1.0 - Y_train) * np.log(1.0 - A))
            cost = (1 / m) * (np.dot(-Y_train.T, np.log(A)) - np.dot((1 - Y_train).T, np.log(1 - A)))

            # Gradient Descent
            # dW = (1 / m) * np.dot(A - Y_train, X_train.T)
            dW = np.dot(X_train.T, (A - Y_train)) / Y_train.size
            # dB = (1 / m) * np.sum(A - Y_train)
            
            W -= self.eta * dW.T
            # B -= self.eta * dB

            # Keeping track of our cost function value
            cost_list.append(cost)
            # if(i%(iter/10000) == 0):
            #     print("cost after ", i, "iteration is : ", cost)


        self.W.append(W)
        self.cost.append(cost_list)
        # return W, B, cost_list

    # 0.5 for z = 0
    # hypothesis function, hθ(z) = sigmoid(θtX), θtX is Matrix Weight Transpose by Matrix X
    # θ is parameter vector For a model containing n features, we have \theta = [\theta_0, \theta_1, ..., \theta_n] containing n + 1 parameters
    def sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    def normalize(self, X):
        return (X - X.mean()) / X.std()

    def print(self):
        for i in range (len(self.houses)) :
            print(self.houses[i], " :\n", self.W[i])

    def save(self, std_deviations, means, filename='./datasets/weights.csv'):
        f = open(filename, 'w+')
        std = list(std_deviations)
        mean = list(means)
        i = 0
        for feat in SELECTED_FEATURES:
            if (i < len(SELECTED_FEATURES) - 1) :
                f.write(f'{feat},')
            else :
                f.write(f'{feat}\n')
            i += 1
        # f.write(f'std,')
        # f.write(f'means\n')
        for each in self.W :
            for i in range (len(each)) :
                if (i < len(SELECTED_FEATURES) - 1) :
                    f.write(f'{each[i]},')
                else :
                    f.write(f'{each[i]}')
            f.write('\n')

        for i in range(len(SELECTED_FEATURES)) :
            if (i < len(SELECTED_FEATURES) - 1) :
                f.write(f'{std[i]},')
            else :
                f.write(f'{std[i]}\n')

        for i in range(len(SELECTED_FEATURES)) :
            if (i < len(SELECTED_FEATURES) - 1) :
                f.write(f'{mean[i]},')
            else :
                f.write(f'{mean[i]}\n')
    # def accuracy(X, Y, W, B):
        # Z = np.dot(W.T, X) + B
        # A = sigmoid(Z)
        # A = A > 0.5
        
        # print(A)
        # A = np.array(A, dtype = 'int64')

    # def visualization(self): # C'est a chier
    #     # i = 0
    #     for house in self.houses :
    #         plt.plot(np.arange(iter), self.cost[0])
    #         plt.title(house)
    #         plt.ylabel("Cost")
    #         plt.xlabel("Itération")
    #         plt.show()
    #         # i += 1


# def accuracy(X, Y, W, B):
#     Z = np.dot(W.T, X) + B
#     A = sigmoid(Z)
#     # A = A > 0.5
    
#     print(A)
#     # A = np.array(A, dtype = 'int64')
    
#     # acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    # print("Accuracy of the model is : ", round(acc, 2), "%")

def normalize(X):
        return X.std(), X.mean(), (X - X.mean()) / X.std()

def clean_dataframe(df):
    # droped = ["Index","Arithmancy","Muggle Studies","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Flying"]
    # features = ["Astronomy","Herbology","Defense Against the Dark Arts","Divination","Ancient Runes","Charms"]

    selected_features = [HOUSES_COL] + [T0_LABEL] + SELECTED_FEATURES
    # df = normalize(df)
    # X_train = df.loc[2:, selected_features]
    std_deviations, means, X_train = normalize(df.loc[:, selected_features[2:]])
    # X_train.insert(0, T0_LABEL, np.ones(X_train.shape[0]))
    X_train.insert(0, "Hogwarts House", df["Hogwarts House"])
    X_train = X_train.dropna()

    Y_train = X_train["Hogwarts House"]
    X_train = X_train.drop("Hogwarts House", axis=1)
    return std_deviations, means, X_train, Y_train

def main(filename):
    # houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    # features = ["Index","Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    try :
        df = pd.read_csv(filename)
    except :
        print("Can't open file")
        return 0

    std_deviations, means, X_train, Y_train = clean_dataframe(df)
    logreg = Log_reg(10000, 0.0015)
    logreg.one_vs_all(X_train, Y_train)
    logreg.save(std_deviations, means)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Input dataset")
    args = parser.parse_args()

    main(args.dataset)