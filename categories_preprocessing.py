import pandas as pd
import numpy as np


def insert_nan_values(column, value):
    # changes all values "value" in given column. example: train = insert_nan_values(train['id'], 0)
    return column.replace([value], np.math.nan)

def preprocess_original_language(data):
    # creating a row for every important language:
    important_languages = ["en", "fr", "hi", "es", "ru", "ja"]

    languages = data["original_language"].unique()
    languages_count = data["original_language"].value_counts()
    languages = np.column_stack((languages, languages_count))

    lang_data = pd.get_dummies(data.original_language)
    lang_data["other_languages"] = 0

    for lang in languages:
        if lang[0] not in important_languages:
            lang_data["other_languages"] += lang_data[lang[0]]
            lang_data = lang_data.drop(columns=[lang[0]])

    data = pd.concat([data, lang_data], axis=1)
    data = data.drop(columns=["original_language"])

    return data


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