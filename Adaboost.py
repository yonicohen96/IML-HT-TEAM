import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import plotly.graph_objects as go

ESTIMATORS_NUM = [1, 3, 5, 7, 10, 25, 50, 75, 100, 150]

# Neria's code to extract basic train and validation data
# TODO after finishing preprocessing, get larger data
train = pd.read_csv('train_capuchon.csv')
train = train.dropna()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train = train.select_dtypes(include=numerics)
y_train = train.drop(["id", "budget", "vote_count", "runtime"], axis=1)
X_train = train.drop(["vote_average", "revenue"], axis=1)
validate = pd.read_csv('validate_capuchon.csv')
validate = validate.dropna()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
validate = validate.select_dtypes(include=numerics)
y_validate = validate.drop(["id", "budget", "vote_count", "runtime"], axis=1)
X_validate = validate.drop(["vote_average", "revenue"], axis=1)

# set two response vectors
Y_trains = [y_train["vote_average"], y_train["revenue"]]
Y_validations = [ y_validate["vote_average"],  y_validate["revenue"]]
scores = [[],[]]

# fit and score
for n_estimators in ESTIMATORS_NUM:
    clf = AdaBoostRegressor(n_estimators=n_estimators, random_state=0, loss="square")
    for prediction_response_ind in range(2):
        clf.fit(X_train, Y_trains[prediction_response_ind])
        score = clf.score(X_validate, Y_validations[prediction_response_ind])
        scores[prediction_response_ind].append(score)


def plot_adaboost():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ESTIMATORS_NUM, y=scores[0],
                             mode='lines',
                             name="vote_average prediction score"))
    fig.add_trace(go.Scatter(x=ESTIMATORS_NUM, y=scores[1],
                             mode='lines',
                             name="revenue score"))
    # general layout
    fig.update_layout(title=f"Adaboost Predicition (Decision Tree with max-depth 1) Score as a Function of Estimators Number",
                      margin=dict(l=200, r=350, t=100, b=70),
                      font=dict(size=18))
    fig.update_yaxes(title="Score")
    fig.update_xaxes(title="Estimators Number")
    fig.show()


if __name__ == '__main__':
    plot_adaboost()