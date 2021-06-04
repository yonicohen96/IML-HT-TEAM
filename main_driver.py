from models import *
import pandas as pd
import pickle
import preprocess
import preprocess_predict

REVENUE_FEATURE_SUBSET = ["vote_count", "budget", "production_companies_other company", "days_passed", "runtime",
                          "month", "genres_Adventure", "genres_Action", "en", "genres_Fantasy", "day_in_week",
                          "genres_Documentary", "production_companies_Paramount", "genres_Family",
                          "production_companies_Universal Pictures", "belongs_to_collection_others", "other_languages",
                          "genres_Drama", "production_companies_Walt Disney Pictures",
                          "production_companies_Warner Bros. Pictures", "production_companies_20th Century Fox",
                          "production_companies_New Line Cinema", "production_companies_Columbia Pictures",
                          "belongs_to_collection_X-Men Collection", "belongs_to_collection_The Dark Knight Collection",
                          "belongs_to_collection_Star Wars Collection", "belongs_to_collection_Harry Potter Collection",
                          "genres_Science Fiction", "genres_Animation"]

VOTE_FEATURE_SUBSET = ["runtime", "vote_count", "genres_Drama", "days_passed", "month", "genres_History",
                       "genres_Comedy", "genres_Horror", "budget", "belongs_to_collection_Halloween Collection",
                       "belongs_to_collection_Star Wars Collection", "production_companies_Warner Bros. Pictures",
                       "genres_Science Fiction", "en", "production_companies_other company",
                       "production_companies_Paramount", "genres_Crime", "genres_Adventure",
                       "production_companies_United Artists", "production_companies_Working Title Films",
                       "belongs_to_collection_The Dark Knight Collection", "production_companies_Universal Pictures",
                       "production_companies_Walt Disney Pictures", "genres_Action", "genres_Fantasy"]

# def get_x_y(df):
#     y_revenue = df['revenue'].to_numpy()
#     y_vote_average = df['vote_average']
#     X = df.drop(["revenue", "vote_average"], axis=1)
#     return X, y_revenue, y_vote_average


if __name__ == '__main__':
    train = pd.read_csv(r'data\train_preprocessed_2200.csv')
    validate = pd.read_csv(r'data\validate_preprocessed_2200.csv')
    test = pd.read_csv(r'test_capuchon.csv')

    # y
    y_revenue_train, y_vote_train = train[["revenue"]], train[["vote_average"]]

    # X
    x_train = train.drop(["revenue", "vote_average"], axis=1)

    # y validate
    d = {'revenue': train.revenue, 'vote_average': train.vote_average}
    y_validate = pd.DataFrame(data=d)
    y_revenue_validate, y_vote_validate = y_validate[["revenue"]], y_validate[["vote_average"]]

    # X validate
    x_validate = train.drop(["revenue", "vote_average"], axis=1)

    revenue_baseline = LinReg()
    revenue_baseline.fit(x_train, y_revenue_train)
    best_revenue_score = revenue_baseline.score(x_validate, y_revenue_validate)

    vote_baseline = LinReg()
    vote_baseline.fit(x_train, y_vote_train)
    best_vote_score = vote_baseline.score(x_validate, y_vote_validate)

    best_revenue_model = revenue_baseline
    best_vote_model = vote_baseline
    lasso_alpha_rev, ridge_alpha_rev = find_alpha_for_lasso_ridge(x_train, y_revenue_train, x_validate,
                                                                  y_revenue_validate)
    lasso_alpha_vote, ridge_alpha_vote = find_alpha_for_lasso_ridge(x_train, y_vote_train, x_validate, y_vote_validate)

    revenue_models = [Lass(lasso_alpha_rev), Lass(lasso_alpha_vote),
                      Forest(depth=20), Forest(), Ridg(ridge_alpha_rev),
                      Ridg(ridge_alpha_vote), Adaboost(100)]
    for m in revenue_models:
        m.fit(x_train, y_revenue_train)
        cur_score = m.score(x_validate, y_revenue_validate)
        if cur_score < best_revenue_score:
            best_revenue_model = m
            best_revenue_score = cur_score

    vote_models = [Lass(lasso_alpha_rev), Lass(lasso_alpha_vote),
                   Forest(depth=20), Forest(depth=15), Ridg(ridge_alpha_rev),
                   Ridg(ridge_alpha_vote), Adaboost(100)]
    for m in vote_models:
        m.fit(x_train, y_vote_train)
        cur_score = m.score(x_validate, y_vote_validate)
        if cur_score < best_vote_score:
            best_vote_model = m
            best_vote_score = cur_score


    with open('models.pkl', 'wb') as f:
        pickle.dump(best_revenue_model, f)
        pickle.dump(best_vote_model, f)