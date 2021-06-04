from models import *
from preprocess import preprocess_main
import preprocess_predict
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle


def get_x_y(df):
    y_revenue = df['revenue'].to_numpy()
    y_vote_average = df['vote_average'].to_numpy()
    X = df.drop(["revenue", "vote_average"], axis=1)
    X = X.to_numpy()
    return X, y_revenue, y_vote_average


if __name__ == '__main__':
    train = pd.read_csv(r'train_capuchon.csv')
    validate = pd.read_csv('data\\validate_capuchon.csv')
    test = pd.read_csv(r'test_capuchon.csv')
    preprocess_train = preprocess_main(train)
    preprocess_validate = preprocess_predict.preprocess_main(validate)
    x_train, y_revenue_train, y_vote_train = get_x_y(preprocess_train)
    x_validate, y_revenue_validate, y_vote_validate = get_x_y(preprocess_validate)
    baseline = linear_reg
    best_revenue_score = baseline(x_train, x_validate, y_revenue_train, y_revenue_validate)
    best_vote_score = baseline(x_train, x_validate, y_vote_train, y_vote_validate)
    best_revenue_model = baseline
    best_vote_model = baseline
    lasso_alpha_rev, ridge_alpha_rev = find_alpha_for_lasso_ridge(x_train, y_revenue_train, x_validate,
                                                                  y_revenue_validate)
    lasso_alpha_vote, ridge_alpha_vote = find_alpha_for_lasso_ridge(x_train, y_vote_train, x_validate, y_vote_validate)
    best_tree_depth(x_train, x_validate, y_revenue_train, y_revenue_validate)  # create graph to choose the params
    best_tree_depth(x_train, x_validate, y_vote_train, y_vote_validate)  # create graph to choose the params

    models = [Lass(lasso_alpha_rev), Lass(lasso_alpha_vote),
              Forest(depth=18), Forest(depth=15), Ridg(ridge_alpha_rev),
              Ridg(ridge_alpha_vote)]
    for m in models:
        m.fit(x_train, y_revenue_train)
        cur_score = m.score(x_validate, y_revenue_validate)
        if cur_score > best_revenue_score:
            best_revenue_model = m

    models = [Lass(lasso_alpha_rev), Lass(lasso_alpha_vote),
              Forest(depth=18), Forest(depth=15), Ridg(ridge_alpha_rev),
              Ridg(ridge_alpha_vote)]
    for m in models:
        m.fit(x_train, y_vote_train)
        cur_score = m.score(x_validate, y_vote_validate)
        if cur_score > best_vote_score:
            best_vote_model = m

    print(best_revenue_model)
    print(best_revenue_score)
    print(best_vote_model)
    print(best_vote_score)

    with open('models.pkl', 'wb') as f:
        pickle.dump(best_revenue_model, f)
        pickle.dump(best_vote_model, f)

    df = pd.read_csv(r"data\train_preprocessed_2200.csv")
    val_df = pd.read_csv(r"data\validate_preprocessed_2200.csv")
