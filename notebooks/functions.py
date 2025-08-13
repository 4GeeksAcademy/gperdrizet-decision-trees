'''Collection of helper functions for notebooks.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def plot_cross_validation(search_results: GridSearchCV, plot_training: bool=False) -> None:
    '''Takes result object from scikit-learn's GridSearchCV(),
    draws plot of hyperparameter set validation score rank vs
    training and validation scores.'''

    results = pd.DataFrame(search_results.cv_results_)
    sorted_results = results.sort_values('rank_test_score')

    plt.title('Hyperparameter optimization')
    plt.xlabel('Hyperparameter set rank')
    plt.ylabel('Validation accuracy (%)')
    plt.gca().invert_xaxis()

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100 + sorted_results['std_test_score']*100,
        sorted_results['mean_test_score']*100 - sorted_results['std_test_score']*100,
        alpha=0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100,
        label='Validation'
    )

    if plot_training:

        plt.fill_between(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score']*100 + sorted_results['std_train_score']*100,
            sorted_results['mean_train_score']*100 - sorted_results['std_train_score']*100,
            alpha=0.5
        )

        plt.plot(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score']*100,
            label='Training'
        )

        plt.legend(loc='best', fontsize='small')

    plt.show()