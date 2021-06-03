import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
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
    # print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % MSE)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(y_val_revenue, y_hat))


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


if __name__ == "__main__":
    revenue_alpha_lasso, revenue_alpha_ridge = \
        find_alpha_for_lasso_ridge(X, y_revenue, X_val, y_val_revenue)
    vote_average_alpha_lasso, vote_average_alpha_ridge = \
        find_alpha_for_lasso_ridge(X, y_vote_average, X_val, y_val_vote_average)