####################################################
# implementation of date column preprocessor phase #
####################################################
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import numpy as np
import plotly as plt
import re
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
    # TODO replace try- except with a condition to check if nan
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        return date.weekday()
    except:
        return np.math.nan


def _get_month_column(row):
    # TODO replace try- except with a condition to check if nan
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        return date.month
    except:
        return np.math.nan

def _get_days_past(row):
    # TODO replace try- except with a condition to check if nan
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        return (dt.datetime.today() - date).days
    except:
        return np.math.nan


def preprocess_date():
    # get data
    df = pd.read_csv(TRAIN_PATH)
    # replace invalid values
    df.loc[df.apply(_date_is_invalid, axis=1), 'release_date'] = np.math.nan
    # add_columns
    df["day_in_week"] = df.apply(_get_day_column, axis=1)
    df["month"] = df.apply(_get_month_column, axis=1)
    df["days_passed"] = df.apply(_get_days_past, axis=1)
    return df


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
    fig.update_xaxes(title="cx")
    fig.show()


if __name__ == '__main__':
    X = preprocess_date()
    print(X[["release_date", "month", "days_passed", "day_in_week"]])

