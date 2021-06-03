divide_data - read the original movies dataset, split randomly to train, validate and test and export them to csv

main_driver- read the train and validate csv, make pre-process (by using next files) and split to design matrix and responses.
It train twice - one for each response, first the baseline (linear regression), then find the best hyper-parameter for each algorithm
- lasso, ridge and Random Forest, and finally pick the best algorithm from them including an option of adaboost.
It saves the "winner" algorithm for each response to pickle.

preprocess - cleaning the data and create new feature to conclude from text and categorial features

preprocess_predict - the same like preprocess, with adjustments to the predict

models - includes the models we try - linear regressin, lasso, ridge and random forest including function to choose hyper-parameter

Random_Forest - object of this model

Adaboost - object of this model

Feature_selection- analysis of corrolations between featuers, useless features and ect.

test_evaluation - read from the pickle and run on the test data - the answers in the pdf.
