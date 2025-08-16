'''Collection of helper functions for notebooks.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


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


def plot_confusion_matrices(models: dict, testing_df: pd.DataFrame) -> None:
    '''Takes dictionary of models and testing dataframe,
    draws confusion matrix for each model.'''

    num_plots = len(models)

    fig, axs = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    axs = axs.flatten()

    fig.suptitle('Test set performance comparison')


    for i, (name, model) in enumerate(models.items()):

        predictions = model.predict(testing_df.drop('Outcome', axis=1))
        accuracy = accuracy_score(predictions, testing_df['Outcome'])*100

        axs[i].set_title(f'{name}\noverall accuracy: {accuracy:.1f}%')
        axs[i].set_xlabel('Predicted outcome')
        axs[i].set_ylabel('True outcome')

        cm = confusion_matrix(testing_df['Outcome'], predictions, normalize='true')

        cm_disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['healthy','diabetic']
        )
        _ = cm_disp.plot(ax=axs[i])

    fig.tight_layout()