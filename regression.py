import pandas as pd
import pickle
import preprocess_predict
import numpy as np

################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    predict_df = pd.read_csv(csv_file)
    if "revenue" in predict_df.columns:
        predict_df = predict_df.drop(["revenue"], axis=1)
    if "vote_average" in predict_df.columns:
        predict_df = predict_df.drop(["vote_average"], axis=1)
    print("before: " + str(predict_df.shape))
    print(predict_df.columns)
    data = preprocess_predict.preprocess_main(predict_df)
    np.array(data.columns)
    with open('models.pkl', 'rb') as file:
        revenue_model = pickle.load(file)
        vote_model = pickle.load(file)
    return list(revenue_model.predict(data)), list(vote_model.predict(data))
