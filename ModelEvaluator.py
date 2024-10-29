import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from scipy.stats import ttest_rel


class ModelEvaluator:
    def __init__(self, model, X_original, Y_original, X_augmented, Y_augmented, X_resampled, Y_resampled, validation_type='holdout',
                 test_size=0.2, random_state=42, n_splits=5, n_repeats=10):
        self.model = model
        self.X_original = X_original
        self.Y_original = Y_original
        self.X_augmented = X_augmented
        self.Y_augmented = Y_augmented
        self.X_resampled = X_resampled
        self.Y_resampled = Y_resampled
        self.validation_type = validation_type
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.results_original = []
        self.results_augmented = []
        self.significance_results = []

    def _calculate_metrics(self, y_true, y_pred, classes):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, labels=classes)
        rec = recall_score(y_true, y_pred, average=None, labels=classes)
        f1 = f1_score(y_true, y_pred, average=None, labels=classes)
        kappa = cohen_kappa_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return acc, prec, rec, f1, kappa, tn, fp, fn, tp

    def _holdout_validation(self, X, Y):
        results = {class_label: [] for class_label in np.unique(Y)}
        for _ in range(self.n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                random_state=self.random_state, stratify=Y)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            classes = np.unique(Y)
            acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(y_test, y_pred, classes)

            for i, class_label in enumerate(classes):
                results[class_label].append({
                    'accuracy': acc,
                    'precision': prec[i],
                    'recall': rec[i],
                    'f1_score': f1[i],
                    'kappa': kappa
                })
        return results

    def _cross_validation(self, X, Y):
        results = {class_label: [] for class_label in np.unique(Y)}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            classes = np.unique(Y)
            acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(y_test, y_pred, classes)

            for i, class_label in enumerate(classes):
                results[class_label].append({
                    'accuracy': acc,
                    'precision': prec[i],
                    'recall': rec[i],
                    'f1_score': f1[i],
                    'kappa': kappa
                })
        return results

    def _leave_one_out(self, X, Y):
        y_true_all = []
        y_pred_all = []

        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            y_true_all.append(y_test[0])
            y_pred_all.append(y_pred[0])

        classes = np.unique(Y)
        acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(np.array(y_true_all), np.array(y_pred_all),
                                                                            classes)

        results = {class_label: [{
            'accuracy': acc,
            'precision': prec[i],
            'recall': rec[i],
            'f1_score': f1[i],
            'kappa': kappa
        }] for i, class_label in enumerate(classes)}

        return results

    def evaluate(self, X, Y):
        if self.validation_type == 'holdout':
            return self._holdout_validation(X, Y)

        elif self.validation_type == 'cross_validation':
            return self._cross_validation(X, Y)

        elif self.validation_type == 'leave_one_out':
            return self._leave_one_out(X, Y)

        else:
            raise ValueError("validation_type must be 'holdout', 'cross_validation' or 'leave_one_out'")

    def _calculate_mean_std(self, results, metrics):
        # Calcula la media y desviación estándar de las métricas por clase
        final_results = []
        for class_label, class_results in results.items():
            summary = {'class': class_label}
            for metric in metrics:
                values = [result[metric] for result in class_results]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
            final_results.append(summary)
        return final_results

    def _perform_significance_test(self, results_original, results_augmented, results_res, metrics, alpha=0.85):
        for class_label in results_original.keys():
            for metric in metrics:
                original_values = [result[metric] for result in results_original[class_label]]
                augmented_values = [result[metric] for result in results_augmented[class_label]]
                resampled_values = [result[metric] for result in results_res[class_label]]

                t_stat_aug, p_value_aug = ttest_rel(original_values, augmented_values)
                significant_aug = 'Yes' if p_value_aug < alpha else 'No'

                self.significance_results.append({
                    'comparison': 'original_vs_augmented',
                    'class': class_label,
                    'metric': metric,
                    't_stat': t_stat_aug,
                    'p_value': p_value_aug,
                    'significant': significant_aug
                })

                t_stat_res, p_value_res = ttest_rel(original_values, resampled_values)
                significant_res = 'Yes' if p_value_res < alpha else 'No'

                self.significance_results.append({
                    'comparison': 'original_vs_resampled',
                    'class': class_label,
                    'metric': metric,
                    't_stat': t_stat_res,
                    'p_value': p_value_res,
                    'significant': significant_res
                })

                t_stat_aug_res, p_value_aug_res = ttest_rel(augmented_values, resampled_values)
                significant_aug_res = 'Yes' if p_value_aug_res < alpha else 'No'

                self.significance_results.append({
                    'comparison': 'augmented_vs_resampled',
                    'class': class_label,
                    'metric': metric,
                    't_stat': t_stat_aug_res,
                    'p_value': p_value_aug_res,
                    'significant': significant_aug_res
                })

    def save_results(self, results_filename='evaluation_results.csv', stats_filename='significance_test_results.csv',
                     alpha=0.85):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa']
        results_original = self.evaluate(self.X_original, self.Y_original)
        results_augmented = self.evaluate(self.X_augmented, self.Y_augmented)
        results_resampled = self.evaluate(self.X_resampled, self.Y_resampled)

        final_results_original = self._calculate_mean_std(results_original, metrics)
        final_results_augmented = self._calculate_mean_std(results_augmented, metrics)
        final_results_resampled = self._calculate_mean_std(results_resampled, metrics)

        self._perform_significance_test(results_original, results_augmented, results_resampled, metrics, alpha)

        df_original = pd.DataFrame(final_results_original)
        df_augmented = pd.DataFrame(final_results_augmented)
        df_resampled = pd.DataFrame(final_results_resampled)
        df_significance = pd.DataFrame(self.significance_results)

        df_original['dataset'] = 'original'
        df_augmented['dataset'] = 'augmented'
        df_resampled['dataset'] = 'resampled'

        df_results = pd.concat([df_original, df_augmented, df_resampled])
        df_results.to_csv(results_filename, index=False)

        df_significance.to_csv(stats_filename, index=False)
