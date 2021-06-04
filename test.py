import regression
import pandas as pd
from sklearn.metrics import mean_squared_error

a, b = regression.predict(r"test_capuchon.csv")
# y = pd.read_csv(r"test_capuchon - responses.csv")
# y_rev = y["revenue"]
# y_vot = y["vote_average"]
print(a)
print(b)
# print(mean_squared_error(y_rev, a))
# print(mean_squared_error(y_vot, b))
