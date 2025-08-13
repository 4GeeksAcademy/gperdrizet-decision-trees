'''Globals for notebooks and modules.'''


##################################################################################
# Files and paths
##################################################################################

# Data
RAW_DATA_FILE='../data/raw/dataset.parquet'
DATA_FILE='../data/processed/dataset.pkl'

# Model hyperparameters
DECISION_TREE_HYPERPARAMETERS='../data/decision_tree_hyperparameters.pkl'
RANDOM_FOREST_HYPERPARAMETERS='../data/random_forest_hyperparameters.pkl'
GRADIENT_BOOSTING_HYPERPARAMETERS='../data/gradient_boosting_hyperparameters.pkl'

# Trained models
DECISION_TREE_MODEL='../models/decision_tree.pkl'
RANDOM_FOREST_MODEL='../models/random_forest.pkl'
GRADIENT_BOOSTING_MODEL='../models/gradient_boosting.pkl'

# Scores
CROSS_VAL_SCORES_FILE='../data/cross_val_scores.pkl'

##################################################################################
# Constants for model training and evaluation
##################################################################################

# Class weight for models that accept it
CLASS_WEIGHT = 'balanced'

# Cross-validation folds for scoring
CROSS_VAL_FOLDS = 5

# Random search iterations
RANDOM_SEARCH_ITERATIONS = 2000