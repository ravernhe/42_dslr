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
#     def binary_classifier(self, X_train, Y_train):
#         m = X_train.shape[1]
#         n = X_train.shape[0]
        
#         W = np.zeros((n,1))
#         B = 0
        
#         cost_list = []
        
#         for i in range(self.iter):
            
#             Z = np.dot(W.T, X) + B
#             A = self.sigmoid(Z)
            
#             # cost function
#             cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
            
#             # Gradient Descent
#             dW = (1/m)*np.dot(A-Y, X.T)
#             dB = (1/m)*np.sum(A - Y)
            
#             W = W - self.eta*dW.T
#             B = B - self.eta*dB
            
#             # Keeping track of our cost function value
#             cost_list.append(cost)
            
#             if(i%(self.iter/10) == 0):
#                 print("cost after ", i, "iteration is : ", cost)
            
#         return W, B, cost_list

#     # 0.5 for z = 0
#     # hypothesis function, hθ(z) = sigmoid(θtX), θtX is Matrix Weight Transpose by Matrix X
#     # θ is parameter vector For a model containing n features, we have \theta = [\theta_0, \theta_1, ..., \theta_n] containing n + 1 parameters
#     def sigmoid(self, z):
#         return (1.0 / (1.0 + np.exp(-z)))

if __name__ == "__main__":
    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    df = df.sort_index()