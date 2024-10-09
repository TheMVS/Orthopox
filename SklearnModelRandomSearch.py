import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from SklearnLoader import SklearnLoader


class SklearnModelRandomSearch:
    def __init__(self, X, Y, models_params=None, n_iter=2, cv=10, test_size=0.1, random_state=42):
        self.X = X
        self.Y = Y
        self.n_iter = n_iter
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state

        # Default models and hyperparameters with class_weight balanced for those that support it
        self.models_params = models_params if models_params else {
            'RandomForest': {
                'clf__n_estimators': [50, 100, 150],
                'clf__max_depth': [5, 10, 15],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__class_weight': ['balanced']  # Class weights balanced
            },
            'SVM': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['linear', 'rbf', 'poly'],
                'clf__gamma': ['scale', 'auto'],
                'clf__class_weight': ['balanced']  # Class weights balanced
            },
            'LogisticRegression': {
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__solver': ['liblinear', 'lbfgs', 'saga'],
                'clf__class_weight': ['balanced']  # Class weights balanced
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

        # Append results to the Excel file incrementally
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=writer.sheets == {})

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
            # Get n_iter best models and their parameters
            for rank in range(self.n_iter):
                best_model = search.cv_results_['params'][rank]
                mean_test_score = search.cv_results_['mean_test_score'][rank]
                std_test_score = search.cv_results_['std_test_score'][rank]

                kappa, f1 = self._evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)

                model_results.append({
                    'model': model_name,
                    'rank': rank + 1,
                    'mean_test_score': mean_test_score,
                    'std_test_score': std_test_score,
                    'best_params': best_model,
                    'kappa': kappa,
                    'f1_score': f1
                })

            # Save results of the current model to the Excel file incrementally
            self._save_partial_results_to_excel(model_results, filename)

    def run_search_and_save(self, filename='random_search_results.xlsx'):
        self.perform_search(filename)
