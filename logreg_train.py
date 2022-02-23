import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            Y = Y.reshape(1, X_train.shape[1])

            X_train = self.normalize(X_train)
            self.binary_classifier(X_train, Y, self.eta, self.iter)
            # print(each, ":")
            # print(W)        

    def binary_classifier(self, X_train, Y_train, eta, iter):
        m = X_train.shape[1]
        n = X_train.shape[0]

        W = np.zeros((n,1))
        B = 0
            
        cost_list = []
        for i in range(self.iter):
                
            Z = np.dot(W.T, X_train) + B
            A = self.sigmoid(Z)
            
            # cost function
            cost = (-(1.0 / m)) * np.sum(Y_train * np.log(A) + (1.0 - Y_train) * np.log(1.0 - A))
            # Gradient Descent
            dW = (1 / m) * np.dot(A - Y_train, X_train.T)
            dB = (1 / m) * np.sum(A - Y_train)
            
            W = W - self.eta * dW.T
            
            B = B - self.eta * dB
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

    def save(self, filename='./datasets/weights.csv'):
        f = open(filename, 'w+')
        features = ["Astronomy","Herbology","Defense Against the Dark Arts","Divination","Ancient Runes","Charms"]
        i = 0
        for feat in features:
            if (i < len(features) - 1) :
                f.write(f'{feat},')
            else :
                f.write(f'{feat}\n')
            i += 1

        for each in self.W :
            j = 0
            for i in range (len(each)) :
                if (j < len(features) - 1) :
                    f.write(f'{each[i]},')
                else :
                    f.write(f'{each[i]}')
                j += 1
            f.write('\n')

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

def clean_dataframe(X_train):
    droped = ["Index","Arithmancy","Muggle Studies","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Flying"]

    X_train = X_train.dropna()
    Y_train = X_train["Hogwarts House"]
    X_train = X_train.select_dtypes("number")

    for each in droped :
        X_train = X_train.drop(each, axis=1)

    X_train = X_train.values.T

    return X_train, Y_train

def main(filename):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    # features = ["Index","Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    try :
        X_train = pd.read_csv(filename)
    except :
        print("Can't open file")
        return 0

    X_train, Y_train = clean_dataframe(X_train)
    logreg = Log_reg(10000, 0.0015)
    logreg.one_vs_all(X_train, Y_train)
    logreg.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Input dataset")
    args = parser.parse_args()

    main(args.dataset)