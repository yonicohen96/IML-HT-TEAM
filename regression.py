import pandas as pd
import pickle
import preprocess_predict
from main_driver import REVENEU_FEATURE_SUBSET, VOTE_FEATURE_SUBSET

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
    rev_data = preprocess_predict.preprocess_main(predict_df, REVENEU_FEATURE_SUBSET)
    vot_data = preprocess_predict.preprocess_main(predict_df, VOTE_FEATURE_SUBSET)
    with open('models.pkl', 'rb') as file:
        revenue_model = pickle.load(file)
        vote_model = pickle.load(file)
    return list(revenue_model.predict(rev_data)), list(vote_model.predict(vot_data))
