import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

cols = ['id', 'belongs_to_collection','budget','genres','homepage','original_title','overview','vote_average','vote_count''production_companies''production_countries','release_date','runtime','spoken_languages','status','tagline','title','keywords','cast','crew','revenue','en','es','fr','he','hi','it','other_languages']
df = pd.read_csv('C:\\Users\\toota\\Documents\\SEM D\\IML\\hackathon\\capuchon\\train_preprocessed.csv', sep=',')
val_df = pd.read_csv('C:\\Users\\toota\\Documents\\SEM D\\IML\\hackathon\\capuchon\\validate_preprocessed.csv', sep=',')

# y
y_revenue = df['revenue'].to_numpy()
y_vote_average = df['vote_average'].to_numpy()

# X
curr_df = df[['budget','vote_count','runtime','en','es','fr','he','hi','it','other_languages']]
X = curr_df.to_numpy()

# y validate
y_val_revenue = val_df['revenue'].to_numpy()
y_val_vote_average = val_df['vote_average'].to_numpy()

# X validate
curr_val_df = val_df[['budget','vote_count','runtime','en','es','fr','he','hi','it','other_languages']]
X_val = curr_val_df.to_numpy()

def linear_reg():
# linerar regression - revenue
    reg = LinearRegression()
    reg.fit(X, y_revenue)
    y_hat = reg.predict(X_val)
    print(X_val.size, y_val_revenue.size, y_hat.size)

    MSE = mean_squared_error(y_val_revenue, y_hat)

    # The coefficients
    print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % MSE)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'% r2_score(y_val_revenue, y_hat))
