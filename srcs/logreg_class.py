import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# class Log_reg():
#     def __init__(self, eta = 0.01, iter = 100, weight=None) -> None:
#         self.eta = eta
#         self.iter = iter
#         self.weight = weight

#     # we use a logarithmic function to represent the cost of logistic regression
#     # X is Data to compute | Y is Data Result
    # def binary_classifier(self, X_train, Y_train):
    #     m = X_train.shape[1]
    #     n = X_train.shape[0]
        
    #     W = np.zeros((n,1))
    #     B = 0
        
    #     cost_list = []
        
    #     for i in range(self.iter):
            
    #         Z = np.dot(W.T, X) + B
    #         A = self.sigmoid(Z)
            
    #         # cost function
    #         cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
            
    #         # Gradient Descent
    #         dW = (1/m)*np.dot(A-Y, X.T)
    #         dB = (1/m)*np.sum(A - Y)
            
    #         W = W - self.eta*dW.T
    #         B = B - self.eta*dB
            
    #         # Keeping track of our cost function value
    #         cost_list.append(cost)
            
    #         if(i%(self.iter/10) == 0):
    #             print("cost after ", i, "iteration is : ", cost)
            
    #     return W, B, cost_list

#     # 0.5 for z = 0
#     # hypothesis function, hθ(z) = sigmoid(θtX), θtX is Matrix Weight Transpose by Matrix X
#     # θ is parameter vector For a model containing n features, we have \theta = [\theta_0, \theta_1, ..., \theta_n] containing n + 1 parameters
#     def sigmoid(self, z):
#         return (1.0 / (1.0 + np.exp(-z)))

def binary_classifier(X_train, Y_train, eta, iter):
    m = X_train.shape[1]
    n = X_train.shape[0]

    W = np.zeros((n,1))
    B = 0
        
    cost_list = []
    for i in range(iter):
            
        Z = np.dot(W.T, X_train) + B
        A = sigmoid(Z)
        
        # cost function
        cost = (-(1.0 / m)) * np.sum(Y_train * np.log(A) + (1.0 - Y_train) * np.log(1.0 - A))
        # Gradient Descent
        dW = (1 / m) * np.dot(A - Y_train, X_train.T)
        dB = (1 / m) * np.sum(A - Y_train)
        
        W = W - eta * dW.T
        
        B = B - eta * dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
            
        # if(i%(iter/10000) == 0):
        #     print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list

def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

def normalize(X):
    return (X - X.mean()) / X.std()

def accuracy(X, Y, W, B):
    
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")

if __name__ == "__main__":
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    features = ["Index","Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    droped = ["Index","Arithmancy"]
    X_train = pd.read_csv("./datasets/dataset_train.csv")
    X_train = X_train.dropna()

    Y_train = X_train["Hogwarts House"]
    X_train = X_train.select_dtypes("number")
    for each in droped :
        X_train = X_train.drop(each, axis=1)
    # X_train = X_train.drop("Transfiguration", axis=1)
    # X_train = X_train.drop("Herbology", axis=1)
    X_train = X_train.values.T

    for each in houses :    
        f = lambda x: 0 if x != each else 1
        Y = Y_train.map(f)

        # print(Y)
        

        Y = Y.values
        Y = Y.reshape(1, X_train.shape[1])

        X_train = normalize(X_train)
        iter = 10000
        W, B, cost_list = binary_classifier(X_train, Y, 0.0015, iter)
        print(W)
        accuracy(X_train, Y, W, B)
        # plt.plot(np.arange(iter), cost_list)
        # plt.title(each)
        # plt.ylabel("Cost")
        # plt.xlabel("Itération")
        # plt.show()
    