from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


class Forest:
    def __init__(self, depth):
        self.model = None
        self.forest = RandomForestRegressor(max_depth=depth, random_state=0)

    def fit(self, x, y):
        self.model = self.forest.fit(x, y)

    def predict(self, x):
        return self.forest.predict(x)

    def score(self, x, y):
        y_predict = self.forest.predict(x)
        return mean_squared_error(y, y_predict)


def best_tree_depth(X_train, y_train, X_validate, y_validate):
    # Train trees on [1,50] tree depths and returns a graph of MSE results
    responses = ["vote_average", "revenue"]
    for response in responses:
        mse = np.zeros(50)
        for a in range(1, 51):
            forest = Forest(a)
            forest.fit(X_train, y_train[response])
            mse[a - 1] = forest.score(X_validate, y_validate[response].to_numpy())

        plt.plot(mse, label="MSE")
        plt.title("MSE of " + response + " as a function of Tree depth")
        plt.xlabel("Tree depth")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()