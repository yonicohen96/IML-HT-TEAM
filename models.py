import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from Random_Forest import Forest

df = pd.read_csv(r"data\train_preprocessed_2200.csv")
val_df = pd.read_csv(r"data\validate_preprocessed_2200.csv")

# y
y_revenue = df['revenue'].to_numpy()
y_vote_average = df['vote_average'].to_numpy()

# X
X = df.drop(["revenue", "vote_average"], axis=1)
X = X.to_numpy()

# y validate
y_val_revenue = val_df['revenue'].to_numpy()
y_val_vote_average = val_df['vote_average'].to_numpy()

# X validate
X_val = val_df.drop(["revenue", "vote_average"], axis=1)
X_val = X_val.to_numpy()

def linear_reg(x_train, x_validate, y_train, y_validate):
# linerar regression - revenue
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_validate)
    MSE = mean_squared_error(y_validate, y_hat)
    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % MSE)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'% r2_score(y_val_revenue, y_hat))


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