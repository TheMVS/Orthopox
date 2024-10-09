import os
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform, expon

from SklearnLoader import SklearnLoader


class SklearnModelRandomSearch:
    def __init__(self, X, Y, n_iter=2, cv=10, test_size=0.1, random_state=42):
        self.X = X
        self.Y = Y
        self.n_iter = n_iter
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state

        # Define statistical distributions for model hyperparameters
        self.models_params = {
            'RandomForest': {
                'clf__n_estimators': randint(50, 200),  # Random integers between 50 and 200
                'clf__max_depth': randint(5, 20),  # Random integers between 5 and 20
                'clf__min_samples_split': randint(2, 10),  # Random integers between 2 and 10
                'clf__min_samples_leaf': randint(1, 5),  # Random integers between 1 and 5
                'clf__class_weight': [None, 'balanced']  # Testing only None and 'balanced'
            },
            'SVM': {
                'clf__C': expon(scale=100),  # Exponential distribution for regularization parameter C
                'clf__kernel': ['linear', 'rbf', 'poly'],  # Kernel types fixed
                'clf__gamma': ['scale', 'auto'],  # Fixed gamma options
                'clf__class_weight': [None, 'balanced']  # None and 'balanced' class weights
            },
            'LogisticRegression': {
                'clf__C': uniform(0.01, 10),  # Uniform distribution between 0.01 and 10 for C
                'clf__solver': ['liblinear', 'lbfgs', 'saga'],  # Solver options
                'clf__class_weight': [None, 'balanced']  # Fixed class weight
            },
            'GradientBoosting': {
                'clf__n_estimators': randint(50, 200),  # Random integers between 50 and 200
                'clf__learning_rate': uniform(0.01, 0.3),  # Learning rate between 0.01 and 0.3
                'clf__max_depth': randint(3, 10)  # Max depth between 3 and 10
            },
            'DecisionTree': {
                'clf__max_depth': randint(5, 20),  # Random integers between 5 and 20
                'clf__min_samples_split': randint(2, 10),  # Random integers between 2 and 10
                'clf__min_samples_leaf': randint(1, 5),  # Random integers between 1 and 5
                'clf__class_weight': [None, 'balanced']  # Fixed class weight
            },
            'KNN': {
                'clf__n_neighbors': randint(1, 30),  # Random integers between 1 and 30
                'clf__weights': ['uniform', 'distance'],  # Fixed weight options
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Fixed algorithm options
            }
        }
        self.results = []

    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return kappa, f1

    def _save_partial_results_to_excel(self, results, filename):
        df = pd.DataFrame(results)

        # Check if the file exists and load previous data if it does
        if os.path.exists(filename):
            # Load the workbook to find the last row
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                workbook = load_workbook(filename)
                sheet = workbook.active
                startrow = sheet.max_row
                df.to_excel(writer, index=False, header=False, startrow=startrow)
        else:
            # If file doesn't exist, create it with headers
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, header=True)

    def perform_search(self, filename='random_search_results.xlsx'):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=self.test_size,
                                                            random_state=self.random_state)

        for model_name, param_distributions in self.models_params.items():
            loader = SklearnLoader(model_name)
            model = loader.get_model()

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            search = RandomizedSearchCV(pipeline, param_distributions, n_iter=self.n_iter, cv=self.cv,
                                        scoring=make_scorer(f1_score, average='weighted'),
                                        random_state=self.random_state, n_jobs=-1, return_train_score=True)

            search.fit(X_train, y_train)

            model_results = []
            for rank in range(self.n_iter):
                best_model = search.cv_results_['params'][rank]
                mean_test_score = search.cv_results_['mean_test_score'][rank]
                std_test_score = search.cv_results_['std_test_score'][rank]

                kappa, f1 = self._evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)

                model_results.append({
                    'model': model_name,
                    'rank': rank + 1,
                    'mean_test_score_cv': mean_test_score,
                    'std_test_score_cv': std_test_score,
                    'best_params': best_model,
                    'kappa': kappa,
                    'f1_score': f1
                })

            # Save results of the current model to the Excel file incrementally
            self._save_partial_results_to_excel(model_results, filename)

    def run_search_and_save(self, filename='random_search_results.xlsx'):
        self.perform_search(filename)
