import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer


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


def add_multi_dummies(data, col_name):
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


def make_dummies(data):
    data = add_dummy(data, "belongs_to_collection")
    data = add_multi_dummies(data, "genres")
    data = add_multi_dummies(data, "production_companies")
    #data = add_multi_dummies(data, "keywords")
    return data


def help1(data, col_name):
    col = data[col_name].dropna()
    for index, row in col.items():
        x = ast.literal_eval(row)
        data.at[index, col_name] = x["name"]
    return data


def help2(data, col_name):
    col = data[col_name].dropna()
    for index, row in col.items():
        x = ast.literal_eval(row)
        lst = list()
        for d in x:
            lst.append(d["name"])
        data.at[index, col_name] = lst
    return data


def make_lists(data):
    data = help1(data, "belongs_to_collection")
    data = help2(data, "genres")
    data = help2(data, "production_companies")
    data = help2(data, "keywords")
    return data


if __name__ == '__main__':
    data = pd.read_csv(r"train_capuchon.csv")
    check_all_dicts(data)
    data = make_lists(data)
    data = make_dummies(data)
    print("shape after make dummies: " + str(data.shape))
