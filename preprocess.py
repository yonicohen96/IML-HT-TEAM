import numpy as np
import pandas as pd

NAN = np.math.nan


# Neria
def preprocess_original_language(data):
    # creating a row for every language with more than 30 movies:
    LANGUAGE_THRESHOLD = 30  # amount of movies a language needs to appear in to have its own column

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
    # df_robust = df.copy()
    for column in ['runtime', 'vote_count', 'budget']:
        data[column] = (data[column] - data[column].median()) / data[column].std()

    return data


def preprocess_main():
    df = pd.read_csv('data\\train_capuchon.csv', sep=',')
    df = preprocess_original_language(df)
    df = preprocess_status(df)
    df = number_columns_preprocess(df)
    df.to_csv('data\\train_preprocessed.csv')


if __name__ == "__main__":
    preprocess_main()
