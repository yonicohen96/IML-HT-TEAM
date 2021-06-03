####################################################
# implementation of date column preprocessor phase #
# main function - preprocess_date                  #
####################################################
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import numpy as np

TRAIN_PATH = "train_capuchon.csv"


def _date_is_invalid(row) -> bool:
    # check if row is in the format %d/%m/%Y
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        present = dt.datetime.now()
        if date < present:
            return False
        else:
            return True
    except:
        return True


def _get_day_column(row):
    date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
    return date.weekday()


def _get_month_column(row):
    date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
    return date.month


def _get_days_past(row):
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        return (dt.datetime.today() - date).days
    except:
        return np.math.nan


def get_median_release_date(X: pd):
    X.dropna()
    median_days_pass = X.median()
    median_date = dt.datetime.today() - dt.timedelta(days=median_days_pass)
    return median_date.strftime("%d/%m/%Y")


def preprocess_date(X: pd.DataFrame):
    """
    gets X - pd DataFrame with column "release_date", replace nulls with median date,
    and adds columns -  "month", "days_passed", "day_in_week"
    :param X: pd DataFrame with column "release_date"
    :return: X - modified
    """
    # replace invalid values with nan
    X.loc[df.apply(_date_is_invalid, axis=1), 'release_date'] = np.math.nan
    # change null values to median in date column.:
    # calculate median:
    X["days_passed"] = X.apply(_get_days_past, axis=1)
    median_date = get_median_release_date(X["days_passed"])
    # replace null with median
    X.loc[X["release_date"].isnull(), 'release_date'] = median_date
    # add_columns
    X["days_passed"] = X.apply(_get_days_past, axis=1)
    X["day_in_week"] = X.apply(_get_day_column, axis=1)
    X["month"] = X.apply(_get_month_column, axis=1)
    return X


def plot(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='markers',
                             name="bla"))
    # general layout
    fig.update_layout(title=f"title",
                      margin=dict(l=200, r=350, t=100, b=70),
                      font=dict(size=18))
    fig.update_yaxes(title="y")
    fig.update_xaxes(title="x")
    fig.show()


if __name__ == '__main__':
    df = pd.read_csv(TRAIN_PATH)
    X = preprocess_date(df)
    print(X[["release_date", "month", "days_passed", "day_in_week"]])
