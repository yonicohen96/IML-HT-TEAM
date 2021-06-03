from models import *
from preprocess import *
import pandas as pd
import numpy as np

if __name__ == '__main__':
    cols = ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'original_title', 'overview', 'vote_average',
            'vote_count''production_companies''production_countries', 'release_date', 'runtime', 'spoken_languages',
            'status', 'tagline', 'title', 'keywords', 'cast', 'crew', 'revenue', 'en', 'es', 'fr', 'he', 'hi', 'it',
            'other_languages']
    df = pd.read_csv('C:\\Users\\toota\\Documents\\SEM D\\IML\\hackathon\\capuchon\\train_preprocessed.csv', sep=',')
    val_df = pd.read_csv('C:\\Users\\toota\\Documents\\SEM D\\IML\\hackathon\\capuchon\\validate_preprocessed.csv',
                         sep=',')

    # y
    y_revenue = df['revenue'].to_numpy()
    y_vote_average = df['vote_average'].to_numpy()

    # X
    curr_df = df[['budget', 'vote_count', 'runtime', 'en', 'es', 'fr', 'he', 'hi', 'it', 'other_languages']]
    X = curr_df.to_numpy()

    # y validate
    y_val_revenue = val_df['revenue'].to_numpy()
    y_val_vote_average = val_df['vote_average'].to_numpy()

    # X validate
    curr_val_df = val_df[['budget', 'vote_count', 'runtime', 'en', 'es', 'fr', 'he', 'hi', 'it', 'other_languages']]
    X_val = curr_val_df.to_numpy()

    models = []
    best_model = 0
    for m in models:
        cur_model = m()
        b_c = boosting(cur_model)
        if cur_model > best_model:
            best_model = cur_model





