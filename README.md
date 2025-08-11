# Decision Trees for Diabetes Prediction

[![Codespaces Prebuilds](https://github.com/4GeeksAcademy/gperdrizet-decision-trees/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/4GeeksAcademy/gperdrizet-decision-trees/actions/workflows/codespaces/create_codespaces_prebuilds)

A comprehensive machine learning project focused on diabetes prediction using decision trees and ensemble methods. This project demonstrates advanced machine learning techniques including decision trees, random forests, gradient boosting, and decision threshold tuning through practical exercises with medical data.

## Project Overview

This project analyzes a diabetes dataset to predict patient outcomes using various tree-based machine learning algorithms. The dataset contains medical measurements and provides hands-on experience with:

- Decision tree classification
- Random forest ensemble methods
- Gradient boosting techniques
- Hyperparameter optimization
- Cross-validation and model evaluation
- Decision threshold tuning
- Data preprocessing and imputation

## Getting Started

### Option 1: GitHub Codespaces (Recommended)

1. **Fork the Repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - 4Geeks students: set 4GeeksAcademy as the owner - 4Geeks pays for your codespace usage. All others, set yourself as the owner
   - Give the fork a descriptive name. 4Geeks students: I recommend including your GitHub username to help in finding the fork if you lose the link
   - Click "Create fork"
   - 4Geeks students: bookmark or otherwise save the link to your fork

2. **Create a GitHub Codespace**
   - On your forked repository, click the "Code" button
   - Select "Create codespace on main"
   - If the "Create codespace on main" option is grayed out - go to your codespaces list from the three-bar menu at the upper left and delete an old codespace
   - Wait for the environment to load (dependencies are pre-installed)

3. **Start Working**
   - Open the MVP notebooks in the [`notebooks`](notebooks) directory to begin with guided exercises
   - Progress to the solution notebooks to see complete implementations

### Option 2: Local Development

1. **Prerequisites**
   - Git
   - Python >= 3.10

2. **Fork the repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - Optional: give the fork a new name and/or description
   - Click "Create fork"

3. **Clone the repository**
   - From your fork of the repository, click the green "Code" button at the upper right
   - From the "Local" tab, select HTTPS and copy the link
   - Run the following commands on your machine, replacing `<LINK>` and `<REPO_NAME>`

   ```bash
   git clone <LINK>
   cd <REPO_NAME>
   ```

4. **Set Up Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Launch Jupyter & start the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

## Project Structure

```
├── .devcontainer/                             # Development container configuration
├── data/                                      # Data storage directory
├── models/                                    # Trained model storage
│
├── notebooks/                                 # Jupyter notebook directory
│   ├── 01.1-decision_tree_mvp.ipynb           # Decision tree basics (MVP)
│   ├── 01.2-decision_tree_solution.ipynb      # Decision tree complete solution
│   ├── 02.1-random_forest_mvp.ipynb           # Random forest basics (MVP)
│   ├── 02.2-random_forest_solution.ipynb      # Random forest complete solution
│   ├── 03.1-gradient_boosting_mvp.ipynb       # Gradient boosting basics (MVP)
│   ├── 03.2-gradient_boosting_solution.ipynb  # Gradient boosting solution
│   ├── 04-decision_threshold_tuning.ipynb     # Decision threshold optimization
│   ├── configuration.py                       # Project configuration settings
│   └── functions.py                           # Utility functions
│
├── .gitignore                                 # Files/directories not tracked by git
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

## Dataset

The dataset contains diabetes prediction data with the following medical features:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Binary classification target (0: No diabetes, 1: Diabetes)

## Learning Objectives

1. **Decision Trees**: Understand tree-based classification algorithms
2. **Ensemble Methods**: Learn random forests and gradient boosting
3. **Hyperparameter Optimization**: Use randomized search for model tuning
4. **Cross-Validation**: Implement proper model evaluation techniques
5. **Data Preprocessing**: Handle missing values with KNN imputation
6. **Model Comparison**: Compare different algorithms and their performance
7. **Decision Thresholds**: Optimize classification thresholds for better performance
8. **Feature Analysis**: Analyze feature importance and correlations

## Technologies Used

- **Python 3.11**: Core programming language
- **Scikit-learn**: Machine learning algorithms and utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive development environment
- **SciPy**: Statistical functions and distributions

## Model Performance

The project implements and compares:
- **Baseline models**: Constant prediction and logistic regression
- **Decision Trees**: Single tree classifiers with hyperparameter optimization
- **Random Forests**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **Threshold Tuning**: Optimization of decision boundaries

Each model includes comprehensive evaluation with cross-validation, confusion matrices, and accuracy metrics.

## Contributing

This is an educational project. Contributions for improving the analysis, adding new algorithms, or enhancing visualizations are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
