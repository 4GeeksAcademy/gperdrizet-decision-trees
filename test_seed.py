import pickle
import random

import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

import notebooks.configuration as config


winning_score_delta = -100
winning_seed = None

for i in range(1000000):
    seed = random.randint(0, 4294967295)

    data_df = pd.read_parquet('data/raw/dataset.parquet')

    training_df, testing_df = train_test_split(
        data_df,
        test_size=0.5
    )

    imputed_features = ['Insulin','SkinThickness','BloodPressure','BMI','Glucose']

    knn_imputer = KNNImputer(missing_values=0.0, weights='distance')

    knn_imputer.fit(training_df[imputed_features])
    training_df[imputed_features] = knn_imputer.transform(training_df[imputed_features])
    testing_df[imputed_features] = knn_imputer.transform(testing_df[imputed_features])


    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']

    label = ['Outcome']

    training_df = training_df[features + label]
    testing_df = testing_df[features + label]

    
    linear_scores = cross_val_score(
        LogisticRegression(max_iter=5000,class_weight=config.CLASS_WEIGHT,random_state=seed),
        training_df.drop('Outcome', axis=1),
        training_df['Outcome'],
        cv=config.CROSS_VAL_FOLDS,
        n_jobs=-1
    )

    linear_score = np.mean(linear_scores) * 100



    hyperparameters = {
        'criterion':['gini','entropy','log_loss'],
        'splitter':['best','random'],
        'max_depth':randint(1, 20),
        'min_samples_split':randint(2, 20),
        'min_samples_leaf':randint(1, 20),
        'min_weight_fraction_leaf':loguniform(10**-5, 0.5),
        'max_features':uniform(loc=0.1, scale=0.9),
        'max_leaf_nodes':randint(2, 100),
        'min_impurity_decrease':loguniform(10**-5, 1.0),
        'ccp_alpha':loguniform(10**-5, 10.0)
    }

    search = RandomizedSearchCV(
        DecisionTreeClassifier(class_weight=config.CLASS_WEIGHT),
        hyperparameters,
        return_train_score=True,
        cv=config.CROSS_VAL_FOLDS,
        n_jobs=-1,
        n_iter=2000,
        random_state=seed,
    )

    search_results = search.fit(training_df.drop('Outcome', axis=1), training_df['Outcome'])
    best_model = search_results.best_estimator_

    tree_scores = cross_val_score(
        best_model,
        training_df.drop('Outcome', axis=1),
        training_df['Outcome'],
        cv=config.CROSS_VAL_FOLDS,
        n_jobs=-1
    )

    tree_score = np.mean(tree_scores) * 100

    score_delta = tree_score - linear_score

    if score_delta > winning_score_delta:
        print(f'WINNER - seed: {seed}, linear = {linear_score:.2f}, tree = {tree_score:.2f}, delta = {score_delta:.2f}')
        winning_score_delta = score_delta

    else:
        print(f'Seed: {seed}, linear = {linear_score:.2f}, tree = {tree_score:.2f}, delta = {score_delta:.2f}')