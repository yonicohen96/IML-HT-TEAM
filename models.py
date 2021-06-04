import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor


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


class LinReg:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        return mean_squared_error(y, y_predict)


class Forest:
    def __init__(self, depth=15, n_estimators=100):
        self.model = None
        self.forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, random_state=0)

    def fit(self, x, y):
        self.model = self.forest.fit(x, y.values.ravel())

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


class Adaboost:
    def __init__(self, n_estimator):
        self.model = AdaBoostRegressor(n_estimators=n_estimator, random_state=0,
                                       loss="square")

    def fit(self, x, y):
        self.model.fit(x, y.values.ravel())

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        return mean_squared_error(y, y_predict)


class Lass:
    def __init__(self, a):
        self.model = None
        self.lasso = Lasso(alpha=a, max_iter=10000, tol=0.1)

    def fit(self, x, y):
        self.model = self.lasso.fit(x, y)

    def predict(self, x):
        return self.lasso.predict(x)

    def score(self, x, y):
        y_predict = self.lasso.predict(x)
        return mean_squared_error(y, y_predict)


class Ridg:
    def __init__(self, a):
        self.model = None
        self.ridge = Ridge(alpha=a, normalize=True)

    def fit(self, x, y):
        self.model = self.ridge.fit(x, y)

    def predict(self, x):
        return self.ridge.predict(x)

    def score(self, x, y):
        y_predict = self.ridge.predict(x)
        return mean_squared_error(y, y_predict)


def find_alpha_for_lasso_ridge(X_train, y_train, X_validate, y_validate):
    """
    :param X_train:
    :param y_train:
    :param X_validate:
    :param y_validate:
    :return: a tuple of 2 alphas - the first for lasso and the second for ridge
    """
    alphas = 10 ** np.linspace(-3, 2, 100)

    lasso_loss = np.zeros(100)
    ridge_loss = np.zeros(100)

    for ind, a in enumerate(alphas):
        # lasso
        lasso = Lasso(alpha=a, max_iter=10000, tol=0.1)
        lasso.fit(X_train, y_train)

        # ridge
        ridge = Ridge(alpha=a, normalize=True)
        ridge.fit(X_train, y_train)

        # mse and reg
        mse = mean_squared_error(y_validate, lasso.predict(X_validate))
        reg = a * np.linalg.norm(lasso.coef_, ord=1)
        lasso_loss[ind] = mse + reg

        mse = mean_squared_error(y_validate, ridge.predict(X_validate))
        reg = a * np.linalg.norm(ridge.coef_, ord=2)
        ridge_loss[ind] = mse + reg

    lasso_a_ind = np.argmin(lasso_loss)
    ridge_a_ind = np.argmin(ridge_loss)

    # # plot
    # plt.figure()
    # # choose on of the following (lasso/ridge)
    # plt.plot(lasso_loss, 'bo-', label=r'lasso loss', color="purple", alpha=0.6, linewidth=3)
    # plt.plot(ridge_loss, 'bo-', label=r'ridge loss', color="pink", alpha=0.6, linewidth=1)
    # plt.xlabel('Lambda index')
    # plt.ylabel(r'$LOSS$')
    # plt.title(r'Evaluate ridge regression with lambdas - vote_average')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()

    return alphas[lasso_a_ind], alphas[ridge_a_ind]

