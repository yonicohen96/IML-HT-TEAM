import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv(r'data/train_preprocessed_2100.csv')
X = train.drop(['revenue', 'vote_average'], axis = 1)
# X.corr().to_csv("cov_out.csv", index=False)
# plt.show(sns.heatmap(X[["budget","vote_count","runtime","en","other_languages","days_passed","day_in_week","month"]].corr().abs()))
