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


def _check_date_validation(row) -> bool:
    # check if row is in the format %d/%m/%Y
    try:
        date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
        present = dt.datetime.now()
        if date < present:
            return True
        else:
            return False
    except ValueError:
        return False


def _get_day_column(row):
    date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
    return date.weekday()


def _get_month_column(row):
    date = dt.datetime.strptime(row.release_date, "%d/%m/%Y")
    return date.month


def preprocess_date():
    # get data
    df = pd.read_csv(TRAIN_PATH)
    # filter only date column and drop nan
    df = df[["revenue", "release_date"]]
    df = df.dropna()
    # check date validation
    df = df[df.apply(_check_date_validation, axis=1)]
    # add_day_column
    df["day"] = df.apply(_get_day_column, axis=1)
    # add_month_column
    df["month"] = df.apply(_get_month_column, axis=1)
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
    df = preprocess_date()
    # only simple scatter plot - doesnt mean anything:
    plot(df["day"], df["revenue"])
    plot(df["month"], df["revenue"])

