# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error
#
#
# class Forest:
#     def __init__(self, depth):
#         self.model = None
#         self.forest = RandomForestRegressor(max_depth=depth, random_state=0)
#
#     def fit(self, x, y):
#         self.model = self.forest.fit(x, y)
#
#     def predict(self, x):
#         return self.forest.predict(x)
#
#     def score(self, x, y):
#         y_predict = self.forest.predict(x)
#         return mean_squared_error(y, y_predict)
