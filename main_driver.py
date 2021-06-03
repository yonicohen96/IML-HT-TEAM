from models import *
from preprocess import *
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

models = [linear_reg]
for m in models:
    m(X, X_val, y_revenue, y_val_revenue)
    m(X, X_val, y_vote_average, y_val_vote_average)
