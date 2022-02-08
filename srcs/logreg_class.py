import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Log_reg():
    def __init__(self, filename) -> None:
        self.filename = filename
    
    # 0.5 for z = 0
    # hypothesis function, hθ(z) = sigmoid(θtX), θtX is Matrix Weight Transpose by Matrix X
    # θ is parameter vector For a model containing n features, we have \theta = [\theta_0, \theta_1, ..., \theta_n] containing n + 1 parameters
    def sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    # we use a logarithmic function to represent the cost of logistic regression
    def cost_function(self):
        pass