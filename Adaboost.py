from sklearn.ensemble import AdaBoostRegressor
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

ESTIMATORS_NUM = [1, 3, 5, 7, 10, 25, 50, 75, 100, 150]
#TODO check best number and update model
ESTIMATOR_DEFAULT = 100


class Adaboost:
    def __init__(self, n_estimator):
        self.model = AdaBoostRegressor(n_estimators=n_estimator, random_state=0,
                                       loss="square")

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        return mean_squared_error(y, y_predict)


# def check_level():
#     # fit and score
#     for n_estimators in ESTIMATORS_NUM:
#         clf = AdaBoostRegressor(n_estimators=n_estimators, random_state=0, loss="square")
#         for prediction_response_ind in range(2):
#             clf.fit(X_train, Y_trains[prediction_response_ind])
#             score = clf.score(X_validate, Y_validations[prediction_response_ind])
#             scores[prediction_response_ind].append(score)


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
