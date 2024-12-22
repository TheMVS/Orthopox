import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from scipy.stats import ttest_rel
from SklearnLoader import SklearnLoader
import Config
from SampleBalancer import SampleBalancer


class MultiModelEvaluator:
    def __init__(self, models, X_original, Y_original, validation_type='holdout',
                 test_size=0.2, random_state=42, n_splits=5, n_repeats=10):
        self.models = models
        self.X_original = X_original
        self.Y_original = Y_original
        self.validation_type = validation_type
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.results_original = []
        self.results_augmented = []
        self.significance_results = []

    def _save_fold_data(self, X_train, y_train, X_test, y_test, dataset, fold, repetition=None):
        folder_name = f"{dataset}"
        os.makedirs(folder_name, exist_ok=True)

        if repetition is not None:  # holdout
            train_filename = os.path.join(folder_name, f"train_rep_{repetition}.csv")
            test_filename = os.path.join(folder_name, f"test_rep_{repetition}.csv")
        else:  #  cross-validation and leave-one-out
            train_filename = os.path.join(folder_name, f"train_fold_{fold}.csv")
            test_filename = os.path.join(folder_name, f"test_fold_{fold}.csv")

        train_data = pd.DataFrame(X_train)
        train_data['label'] = y_train
        train_data.to_csv(train_filename, index=False)

        test_data = pd.DataFrame(X_test)
        test_data['label'] = y_test
        test_data.to_csv(test_filename, index=False)

    def _calculate_metrics(self, y_true, y_pred, classes):
        acc_per_class = []
        prec = precision_score(y_true, y_pred, average=None, labels=classes)
        rec = recall_score(y_true, y_pred, average=None, labels=classes)
        f1 = f1_score(y_true, y_pred, average=None, labels=classes)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        tn = np.zeros(len(classes))
        fp = np.zeros(len(classes))
        fn = np.zeros(len(classes))
        tp = np.zeros(len(classes))

        for i, cls in enumerate(classes):
            tp[i] = cm[i, i]
            fn[i] = np.sum(cm[i, :]) - tp[i]
            fp[i] = np.sum(cm[:, i]) - tp[i]
            tn[i] = np.sum(cm) - (tp[i] + fn[i] + fp[i])

            # Accuracy por clase: (TP) / Total para la clase
            acc_per_class.append(tp[i] + fn[i]/ tp[i] + fn[i] + fp[i] + tn[i])

        return acc_per_class, prec, rec, f1, kappa, tn, fp, fn, tp

    def _apply_pca(self, X_train, X_test, X_final_test):
        pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_final_test_pca = pca.transform(X_final_test)
        return X_train_pca, X_test_pca, X_final_test_pca

    def _holdout_validation(self, X, Y, X_final_test, Y_final_test, loader, dataset):

        results = {class_label: [] for class_label in np.unique(Y)}
        final_results = {class_label: [] for class_label in np.unique(Y)}

        original_X_final_test = X_final_test
        for counter in range(self.n_repeats):
            print(f"CV iteration {counter + 1} of {dataset} dataset\n")
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                random_state=self.random_state, stratify=Y)

            if dataset == 'augmented':
                X_aug, y_train, classes_aug = loader.data_augmentation(X_train, y_train,
                                                                       prob_flip_horizontal=Config.PROB_FLIP_HORIZONTAL,
                                                                       prob_flip_vertical=Config.PROB_FLIP_VERTICAL,
                                                                       prob_blur=Config.PROB_BLUR,
                                                                       blur_size=Config.BLUR_SIZE)
                X_train = loader.image_to_model_features(X_aug)
            elif dataset == 'balanced':
                balancer = SampleBalancer(random_state=Config.SEED)
                X_train = loader.image_to_model_features(X_train)
                X_res, y_train = balancer.balance(X_train, y_train, method=Config.RESAMPLE_METHOD,
                                                  technique=Config.RESAMPLE_TECHNIQUE)
            else:
                X_train = loader.image_to_model_features(X_train)

            X_test = loader.image_to_model_features(X_test)

            if Config.ATRIBUTE_REDUCTION == 'pca':
                X_train, X_test, X_final_test = self._apply_pca(X_train, X_test, original_X_final_test)

            test_data_folder = "final_test_data"
            os.makedirs(test_data_folder, exist_ok=True)
            test_data = pd.DataFrame(X_final_test)
            test_data['label'] = Y_final_test
            test_data.to_csv(os.path.join(test_data_folder, 'final.csv'), index=False)

            self._save_fold_data(X_train, y_train, X_test, y_test, dataset, fold=None, repetition=counter)


            unique_classes = np.unique(Y)
            train_counts = np.bincount(y_train, minlength=len(unique_classes))
            test_counts = np.bincount(y_test, minlength=len(unique_classes))

            print(f"Class distribution in train set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, train_counts):
                print(f"Class {cls}: {count}")

            print(f"Class distribution in test set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, test_counts):
                print(f"Class {cls}: {count}")

            for model_name in self.models.keys():
                skLoader = SklearnLoader(model_name=model_name, model_params=self.models[model_name])
                model = skLoader.model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                classes = np.unique(Y)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(y_test, y_pred, classes)

                for i, class_label in enumerate(classes):
                    results[class_label].append({
                        'dataset': dataset,
                        'model': model_name,
                        'fold_repetition': f'repetition_{counter}',
                        'accuracy': acc[i],
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })

                y_pred = model.predict(X_final_test)
                classes = np.unique(Y)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(Y_final_test, y_pred, classes)
                for i, class_label in enumerate(classes):
                    final_results[class_label].append({
                        'dataset': 'final',
                        'model': model_name,
                        'fold_repetition': f'repetition_{counter}',
                        'accuracy': acc[i],
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })

        return results, final_results

    def _cross_validation(self, X, Y, X_final_test, Y_final_test, loader, dataset):
        results = {class_label: [] for class_label in np.unique(Y)}
        final_results = {class_label: [] for class_label in np.unique(Y)}

        original_X_final_test = X_final_test
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        counter = 0
        for train_index, test_index in skf.split(X, Y):
            counter += 1
            print(f"CV iteration {counter} of {dataset} dataset\n")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            if dataset == 'augmented':
                X_aug, y_train, classes_aug = loader.data_augmentation(X_train, y_train,
                                                                       prob_flip_horizontal=Config.PROB_FLIP_HORIZONTAL,
                                                                       prob_flip_vertical=Config.PROB_FLIP_VERTICAL,
                                                                       prob_blur=Config.PROB_BLUR,
                                                                       blur_size=Config.BLUR_SIZE)
                X_train = loader.image_to_model_features(X_aug)
            elif dataset == 'balanced':
                balancer = SampleBalancer(random_state=Config.SEED)
                X_train = loader.image_to_model_features(X_train)
                X_train, y_train = balancer.balance(X_train, y_train, method=Config.RESAMPLE_METHOD,
                                                  technique=Config.RESAMPLE_TECHNIQUE)
            else:
                X_train = loader.image_to_model_features(X_train)

            X_test = loader.image_to_model_features(X_test)

            if Config.ATRIBUTE_REDUCTION == 'pca':
                X_train, X_test, X_final_test = self._apply_pca(X_train, X_test, original_X_final_test)

            test_data_folder = "final_test_data"
            os.makedirs(test_data_folder, exist_ok=True)
            test_data = pd.DataFrame(X_final_test)
            test_data['label'] = Y_final_test
            test_data.to_csv(os.path.join(test_data_folder, 'final.csv'), index=False)

            self._save_fold_data(X_train, y_train, X_test, y_test, dataset, fold=counter)

            unique_classes = np.unique(Y)
            train_counts = np.bincount(y_train, minlength=len(unique_classes))
            test_counts = np.bincount(y_test, minlength=len(unique_classes))

            print(f"Class distribution in train set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, train_counts):
                print(f"Class {cls}: {count}")

            print(f"Class distribution in test set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, test_counts):
                print(f"Class {cls}: {count}")

            for model_name in self.models.keys():
                skLoader = SklearnLoader(model_name=model_name, model_params=self.models[model_name])
                model = skLoader.model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                classes = np.unique(Y)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(y_test, y_pred, classes)
    
                for i, class_label in enumerate(classes):
                    results[class_label].append({
                        'dataset': dataset,
                        'model': model_name,
                        'fold_repetition': f'fold{counter}',
                        'accuracy': acc[i],
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })
    
                y_pred = model.predict(X_final_test)
                classes = np.unique(Y)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(Y_final_test, y_pred, classes)
                for i, class_label in enumerate(classes):
                    final_results[class_label].append({
                        'dataset': 'final',
                        'model': model_name,
                        'fold_repetition': f'fold{counter}',
                        'accuracy': acc[i],
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })

        return results, final_results

    def _leave_one_out(self, X, Y, X_final_test, Y_final_test, loader, dataset):
        results = {class_label: [] for class_label in np.unique(Y)}
        final_results = {class_label: [] for class_label in np.unique(Y)}

        loo = LeaveOneOut()
        counter = 0
        original_X_final_test = X_final_test
        for train_index, test_index in loo.split(X):
            counter += 1
            print(f"LOO iteration {counter} of {dataset} dataset\n")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            if Config.ATRIBUTE_REDUCTION == 'pca':
                X_train, X_test, X_final_test = self._apply_pca(X_train, X_test, original_X_final_test)

            test_data_folder = "final_test_data"
            os.makedirs(test_data_folder, exist_ok=True)
            test_data = pd.DataFrame(X_final_test)
            test_data['label'] = Y_final_test
            test_data.to_csv(os.path.join(test_data_folder, 'final.csv'), index=False)

            unique_classes = np.unique(Y)
            train_counts = np.bincount(y_train, minlength=len(unique_classes))
            test_counts = np.bincount(y_test, minlength=len(unique_classes))

            print(f"Class distribution in train set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, train_counts):
                print(f"Class {cls}: {count}")

            print(f"Class distribution in test set of {dataset} dataset:\n")
            for cls, count in zip(unique_classes, test_counts):
                print(f"Class {cls}: {count}")

            for model_name in self.models.keys():
                skLoader = SklearnLoader(model_name=model_name, model_params=self.models[model_name])
                model = skLoader.model
                model.fit(X_train, y_train)

                # Predictions for LOO fold
                y_pred = model.predict(X_test)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(y_test, y_pred, unique_classes)

                for i, class_label in enumerate(unique_classes):
                    results[class_label].append({
                        'dataset': dataset,
                        'model': model_name,
                        'fold_repetition': f'loo{counter}',
                        'class': class_label,
                        'accuracy': acc,
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })

                # Final test predictions
                y_pred_final = model.predict(X_final_test)
                acc, prec, rec, f1, kappa, tn, fp, fn, tp = self._calculate_metrics(Y_final_test, y_pred_final,
                                                                                    unique_classes)

                for i, class_label in enumerate(unique_classes):
                    final_results[class_label].append({
                        'dataset': 'final',
                        'model': model_name,
                        'fold_repetition': f'loo{counter}',
                        'class': class_label,
                        'accuracy': acc,
                        'precision': prec[i],
                        'recall': rec[i],
                        'f1_score': f1[i],
                        'kappa': kappa
                    })

        return results, final_results

    def evaluate(self, X, Y, X_test, Y_test, loader, dataset='original'):
        if self.validation_type == 'holdout':
            return self._holdout_validation(X, Y, X_test, Y_test, loader, dataset)
        elif self.validation_type == 'cross_validation':
            return self._cross_validation(X, Y, X_test, Y_test, loader, dataset)

        elif self.validation_type == 'leave_one_out':
            return self._leave_one_out(X, Y, X_test, Y_test, loader, dataset)

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

    def save_results(self, loader, results_filename='evaluation_results.csv',
                     stats_filename='significance_test_results.csv',
                     alpha=0.85):
        X, X_test, Y, Y_test = train_test_split(self.X_original, self.Y_original, test_size=self.test_size,
                                                random_state=self.random_state, stratify=self.Y_original)
        X_test = loader.image_to_model_features(X_test)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa']
        results_original, final_original = self.evaluate(X, Y, X_test, Y_test, loader, 'original')
        results_augmented, final_augmented = self.evaluate(X, Y, X_test, Y_test, loader, 'augmented')
        results_resampled, final_resampled = self.evaluate(X, Y, X_test, Y_test, loader, 'balanced')

        all_results = []
        for dataset_results in [results_original, results_augmented, results_resampled]:
            for class_label, records in dataset_results.items():
                for record in records:
                    structured_record = {'class': class_label, **record}
                    all_results.append(structured_record)

        # Guardar el archivo CSV con la clase como la primera columna
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(by=['class'])
        df_results.to_csv(results_filename, index=False)

        # Procesar resultados finales
        all_final_results = []
        for final_dataset_results in [final_original, final_augmented, final_resampled]:
            for class_label, records in final_dataset_results.items():
                for record in records:
                    structured_record = {'class': class_label, **record}
                    all_final_results.append(structured_record)

        df_final_results = pd.DataFrame(all_final_results)
        df_final_results = df_final_results.sort_values(by=['class'])
        df_final_results.to_csv('final_' + results_filename, index=False)

