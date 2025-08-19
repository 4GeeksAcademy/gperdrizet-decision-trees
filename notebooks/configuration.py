'''Globals for notebooks and modules.'''

from sklearn.model_selection import StratifiedShuffleSplit

# WINNER - seed: 2271137437, linear = 74.19, tree = 78.65, delta = 4.46
RANDOM_SEED = 2271137437

##################################################################################
# Files and paths
##################################################################################

# Data
RAW_DATA_FILE = '../data/raw/dataset.parquet'
DATA_FILE = '../data/processed/dataset.pkl'

# File for list of imputed features
IMPUTED_FEATURES_FILE = '../models/imputed_features.pkl'

# Model hyperparameters
DECISION_TREE_HYPERPARAMETERS = '../models/decision_tree_hyperparameters.pkl'
RANDOM_FOREST_HYPERPARAMETERS = '../models/random_forest_hyperparameters.pkl'
GRADIENT_BOOSTING_HYPERPARAMETERS = '../models/gradient_boosting_hyperparameters.pkl'

# Trained models
LOGISTIC_REGRESSION_MODEL = '../models/logistic_regression.pkl'
DECISION_TREE_MODEL = '../models/decision_tree.pkl'
RANDOM_FOREST_MODEL = '../models/random_forest.pkl'
GRADIENT_BOOSTING_MODEL = '../models/gradient_boosting.pkl'
FINAL_MODEL = '../models/final_model.pkl'

# Scores
CROSS_VAL_SCORES_FILE = '../data/cross_val_scores.pkl'

##################################################################################
# Constants for model training and evaluation
##################################################################################

# Class weight for models that accept it
CLASS_WEIGHT = 'balanced'

# Cross validation strategy
CV_FOLDS = 10
CV_SPLIT = 0.5

CROSS_VAL = StratifiedShuffleSplit(
        n_splits=CV_FOLDS,
        test_size=CV_SPLIT,
        random_state=RANDOM_SEED
    )

# Random search iterations for hyperparameter tuning
RANDOM_SEARCH_ITERATIONS = 40000