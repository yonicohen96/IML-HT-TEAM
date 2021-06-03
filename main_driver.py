from models import *
from preprocess import preprocess_main
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from Random_Forest import Forest
import pickle

def read_data():

def split_data():

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
    preprocess_validate = preprocess_main(validate)
    x_train, y_revenue_train, y_vote_train = get_x_y(preprocess_train)
    x_validate, y_revenue_validate, y_vote_validate = get_x_y(preprocess_train)
    baseline = linear_reg
    best_revenue_score = baseline(x_train, x_validate, y_revenue_train, y_revenue_validate)
    best_vote_score = baseline(x_train, x_validate, y_vote_train, y_vote_validate)
    best_revenue_model = baseline
    best_vote_model = baseline
    lasso_alpha_rev, ridge_alpha_rev = find_alpha_for_lasso_ridge(x_train, y_revenue_train, x_validate, y_revenue_validate)
    lasso_alpha_vote, ridge_alpha_vote = find_alpha_for_lasso_ridge(x_train, y_vote_train, x_validate, y_vote_validate)
    best_tree_depth(x_train, x_validate, y_revenue_train, y_revenue_validate) # create graph to choose the params
    best_tree_depth(x_train, x_validate, y_vote_train, y_vote_validate)  # create graph to choose the params
    forest_revenue_k, forest_revenue_T = 10, 10
    forest_vote_k, forest_vote_T = 10, 10
    models = [Lasso(lasso_alpha_rev), Lasso(alpha=lasso_alpha_vote, max_iter=10000, tol=0.1),
              ridge, forest]
    for m in models:
        cur_revenue = m(x_train, x_validate, y_revenue_train, y_revenue_validate)
        cur_vote = m(x_train, x_validate, y_vote_train, y_vote_validate)
        if cur_revenue > best_revenue_score:
            best_revenue_model = m
        if cur_vote > best_vote_model:
            best_vote_model = m
    with open('models.pkl', 'wb') as f:
        pickle.dump(best_revenue_model)
        pickle.dump(best_vote_model)





    df = pd.read_csv(r"data\train_preprocessed_2200.csv")
    val_df = pd.read_csv(r"data\validate_preprocessed_2200.csv")


