def get_relevant_features(X, y, cutoff=0.05):
    # cutoff should be a value between [0,1]. 0 is no correlation, 1 is maximal correlation
    MI = mutual_info_regression(X, y, n_neighbors=10)
    irrelevant_cols = []
    relevant_cols = []
    for i in range(len(X.columns.values)):
        if MI[i] > cutoff:
            relevant_cols.append(X.columns.values[i])
        else:
            irrelevant_cols.append(X.columns.values[i])
    return relevant_cols, irrelevant_cols