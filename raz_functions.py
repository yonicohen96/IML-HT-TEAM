import numpy as np
import pandas as pd
import json
import ast


def check_dict(dict, id_name, value_name, res):
    if dict[id_name] in res:
        res[dict[id_name]].add(dict[value_name])
    else:
        res[dict[id_name]] = set()
        res[dict[id_name]].add(dict[value_name])

def dict_status(ids):
    print("different ids count: " + str(len(ids)))
    duplicates = 0
    for key in ids:
        if len(ids[key])>1:
            duplicates+=1
    print("there are " + str(duplicates) + " ids with different values")


def check_dict_id_value(col, id_name, value_name):
    col = col.dropna()
    ids = dict()
    for cell in col:
        check_dict(ast.literal_eval(cell), id_name, value_name, ids)
    dict_status(ids)


def check_list_id_value(col, id_name, value_name):
    col = col.dropna()
    ids = dict()
    for cell in col:
        x = ast.literal_eval(cell)
        for d in x:
            check_dict(d, id_name, value_name, ids)
    dict_status(ids)


def check_all_dicts(data):
    check_dict_id_value(data["belongs_to_collection"], "id", "name")
    check_list_id_value(data["genres"], "id", "name")
    check_list_id_value(data["production_companies"], "id", "name")
    check_list_id_value(data["keywords"], "id", "name")
    #check_list_id_value(data["Cast"], "id", "name")
    #check_list_id_value(data["Crew"], "id", "name")


if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\Raz\Study\Computer_Science\IML\repo\IML.HUJI\Hackathon\playground\movies_dataset.csv")
    check_all_dicts(data)