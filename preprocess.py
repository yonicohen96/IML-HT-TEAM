import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json
import re
import datetime as dt

NAN = np.math.nan
LANGUAGE_THRESHOLD = 30  # amount of movies a language needs to appear in to have its own column
LEADING_20_COMPANIES = ["Working Title Films","Village Roadshow Pictures",
                        "Sony Pictures", "Relativity Media","DreamWorks Pictures","United Artists","StudioCanal",
                        "Lionsgate","TriStar Pictures","Canal+","Miramax","Touchstone Pictures",
                        "Walt Disney Pictures","New Line Cinema","Metro-Goldwyn-Mayer","20th Century Fox",
                        "Paramount","Columbia Pictures","Warner Bros. Pictures","Universal Pictures"]

# Raz
def add_multi_dummies(data, col_name):
    data.loc[data[col_name].isna(), col_name] = "Unclassified"
    #col = data[col_name].dropna()
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(data[col_name])
    names = [col_name + "_" + name for name in mlb.classes_]
    data = pd.concat([data, pd.DataFrame(encoded, columns=names, index=data.index)], axis=1)
    data = data.drop([col_name], axis=1)
    return data


def add_dummy(data, col_name):
    encoded = pd.get_dummies(data[col_name], prefix=col_name)
    data = pd.concat([data, encoded], axis=1)
    data = data.drop([col_name], axis=1)
    return data


def make_nan(data, columns):
    for col in columns:
        data.loc[data[col] == "[]", [col]] = np.math.nan
    return data


def validate_single_dict(s):
    pattern = r'{(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")),)*(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")))}'
    check = re.compile(pattern)
    if check.match(s) is None:
        return False
    return True


def validate_multi_dict(s):
    pattern = r'\[({(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")),)*(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")))},\s*)*({(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")),)*(\s*("[^"]*":)((\s*\d*)|(\s"[^"]*")))})\]'
    check = re.compile(pattern)
    if check.match(s) is None:
        return False
    return True


def parser_col_single(data, col_name):
    col_df = data[col_name].dropna()
    col_df = col_df[col_df.notnull()]
    col_df = col_df.apply(lambda s: str(s).replace('\'', '"'))
    for index, cell in col_df.items():
        if validate_single_dict(cell):
            try:
                data.at[index, col_name] = json.loads(cell)
            except:
                data.at[index, col_name] = np.nan
        else:
            data.at[index, col_name] = np.nan
    return data


def parser_col_multi(data, col_name):
    col_df = data[col_name].dropna()
    col_df = col_df[col_df.notnull()]
    col_df = col_df.apply(lambda s: str(s).replace('\'', '"'))
    for index, cell in col_df.items():
        if validate_multi_dict(cell):
            try:
                data.at[index, col_name] = json.loads(cell)
            except:
                data.at[index, col_name] = np.nan
        else:
            data.at[index, col_name] = np.nan
    return data


def names_list(data, col_name, value):
    for index, cell in data[col_name].items():
        if type(cell) == dict:
            data.at[index, col_name] = cell[value]
        elif type(cell) == list:
            lst = list()
            for d in cell:
                lst.append(d[value])
            data.at[index, col_name] = lst
        else:
            data.at[index, col_name] = np.math.nan
    return data


def _get_leading_company(row):
    if type(row.production_companies) == list:
        for company in row.production_companies:
            if company in LEADING_20_COMPANIES:
                return company
    return "other company"


def parser_dicts(data):
    cols1 = ["belongs_to_collection"]
    cols2 = ["genres", "production_companies", "production_countries", "spoken_languages", "keywords", "cast", "crew"]
    data = make_nan(data, cols1 + cols2)
    for col in cols1:
        data = parser_col_single(data, col)
    for col in cols2:
        data = parser_col_multi(data, col)
    only_names = ["belongs_to_collection", "genres", "production_companies", "production_countries", "keywords"]
    for col in only_names:
        data = names_list(data, col, "name")
    data = names_list(data, "spoken_languages", "english_name")
    data = pre_belongs_to_collection(data)
    data = add_multi_dummies(data, "genres")
    # create dummies based on "production_companies"
    #TODO check if works
    data["production_companies"] = data.apply(_get_leading_company, axis=1)
    data = add_dummy(data, "production_companies")
    return data


def pre_belongs_to_collection(data):
    num_rows = data.shape[0]
    collection_counts = data["belongs_to_collection"].value_counts()[:30]
    best_collections = np.array(collection_counts.index)
    np.savetxt(r"\data\collection_features.csv", best_collections, delimiter=",")
    for collection in best_collections:
        name = "belongs_to_collection_" + collection
        data[name] = np.zeros(num_rows)
    others = "belongs_to_collection_others"
    data[others] = np.zeros(num_rows)
    for index, cell in data["belongs_to_collection"].items():
        if type(cell) != str:
            continue
        new_name = "belongs_to_collection_" + cell
        if new_name in data.columns:
            data.at[index, new_name] = 1
        else:
            data.at[index, others] = 1
    data = data.drop(["belongs_to_collection"], axis=1)
    return data


# Neria
def preprocess_original_language(data):
    # creating a row for every language with more than 30 movies:
    languages = data["original_language"].unique()
    languages_count = data["original_language"].value_counts()
    languages = np.column_stack((languages, languages_count))

    lang_data = pd.get_dummies(data.original_language)
    lang_data["other_languages"] = 0
    for lang in languages:
        if lang[1] < LANGUAGE_THRESHOLD:
            lang_data["other_languages"] += lang_data[lang[0]]
            lang_data = lang_data.drop(columns=[lang[0]])

    data = pd.concat([data, lang_data], axis=1)
    data = data.drop(columns=["original_language"])

    return data


def preprocess_status(data):
    return data.drop((data[data["status"] != "Released"]).index)


# Toot
def number_columns_preprocess(data):
    # runtime - change all 0 and nan values to the median
    data.loc[data['runtime'] == 0, 'runtime'] = np.math.nan
    data.loc[data['runtime'].isna(), 'runtime'] = np.math.nan
    data.loc[data['runtime'].isna(), 'runtime'] = data['runtime'].median()

    # vote_count - no changes

    # budget - change all <= 5000 and nan values to the median
    data.loc[data['budget'] <= 5000, 'budget'] = np.math.nan
    data.loc[data['budget'].isna(), 'budget'] = np.math.nan
    data.loc[data['budget'].isna(), 'budget'] = data['budget'].median()

    # normalize the runtime, vote_count, budget
    for column in ['runtime', 'vote_count', 'budget']:
        data[column] = (data[column] - data[column].median()) / data[column].std()

    return data


# release_date column functions:

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


def _get_median_release_date(X: pd):
    X.dropna()
    median_days_pass = X.median()
    median_date = dt.datetime.today() - dt.timedelta(days=median_days_pass)
    return median_date.strftime("%d/%m/%Y")


# TODO check if preprocess_date function works also here

def preprocess_date(X: pd.DataFrame):
    """
    gets X - pd DataFrame with column "release_date", replace nulls with median date,
    and adds columns -  "month", "days_passed", "day_in_week"
    :param X: pd DataFrame with column "release_date"
    :return: X - modified
    """
    # replace invalid values with nan
    X.loc[X.apply(_date_is_invalid, axis=1), 'release_date'] = np.math.nan
    # change null values to median in date column.:
    # calculate median:
    X["days_passed"] = X.apply(_get_days_past, axis=1)
    median_date = _get_median_release_date(X["days_passed"])
    # replace null with median
    X.loc[X["release_date"].isnull(), 'release_date'] = median_date
    # add_columns
    X["days_passed"] = X.apply(_get_days_past, axis=1)
    X["day_in_week"] = X.apply(_get_day_column, axis=1)
    X["month"] = X.apply(_get_month_column, axis=1)
    return X


def preprocess_main():
    df = pd.read_csv('data\\validate_capuchon.csv', sep=',')
    df = preprocess_original_language(df)
    df = preprocess_status(df)
    df = number_columns_preprocess(df)
    df = preprocess_date(df)
    # df = parser_dicts(df)
    df.to_csv('data\\validate_preprocessed.csv', index=False)


if __name__ == "__main__":
    preprocess_main()