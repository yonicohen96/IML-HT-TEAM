import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json
import re

"""
def check_dict(dict, id_name, value_name, res):
    if dict[id_name] in res:
        res[dict[id_name]].add(dict[value_name])
    else:
        res[dict[id_name]] = set()
        res[dict[id_name]].add(dict[value_name])


def dict_status(ids, col_name):
    print(col_name)
    print("different ids count: " + str(len(ids)))
    duplicates = 0
    for key in ids:
        if len(ids[key])>1:
            duplicates+=1
    print("there are " + str(duplicates) + " ids with different values")
    print()


def check_dict_id_value(col, id_name, value_name, col_name):
    col = col.dropna()
    ids = dict()
    for cell in col:
        check_dict(ast.literal_eval(cell), id_name, value_name, ids)
    dict_status(ids, col_name)


def check_list_id_value(col, id_name, value_name, col_name):
    col = col.dropna()
    ids = dict()
    for cell in col:
        x = ast.literal_eval(cell)
        for d in x:
            check_dict(d, id_name, value_name, ids)
    dict_status(ids, col_name)


def check_all_dicts(data):
    check_dict_id_value(data["belongs_to_collection"], "id", "name", "belongs_to_collection")
    check_list_id_value(data["genres"], "id", "name", "genres")
    check_list_id_value(data["production_companies"], "id", "name", "production_companies")
    check_list_id_value(data["keywords"], "id", "name", "keywords")
"""


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
    #data = add_dummy(data, "belongs_to_collection")
    #data = add_multi_dummies(data, "genres")
    #data = add_multi_dummies(data, "keywords")
    return data


if __name__ == '__main__':
    data = pd.read_csv(r"train_capuchon.csv")
    data = parser_dicts(data)
    # for cell in data["genres"]:
    #     if type(cell) == str:
    #         print(cell)
    #     elif type(cell) == list:
    #         print(cell)
    #parser(data, cols)
    #data = remove_chars(data, cols)
    #data = make_nan(data, cols)
    #check_all_dicts(data)
    #data = make_lists(data)
    # data["genres"].value_counts().to_csv("raz_check2.csv")
    # print(data["genres"].value_counts())
    # print(data["genres"].isna().sum())
    #data = make_dummies(data)
    print("shape after make dummies: " + str(data.shape))
