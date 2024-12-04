SEED = 42

BASE_PATH = "./data"
DATA_SUMMARY_FILE = 'data_summary.txt'
RANDOMSEARCH_FILE = 'random_search_results'
RANDOMSEARCH_FORMAT = 'csv'
RESULTS_FILE = 'evaluation_results.csv'
STATS_FILE = 'custom_significance_results.csv'

IMAGE_SIZE= (224, 224, 3)

DATA_AUGMENTATION = False
PROB_FLIP_HORIZONTAL = 0.0
PROB_FLIP_VERTICAL = 0.5
PROB_BLUR = 0.5
BLUR_SIZE = 5

RESAMPLE_METHOD = "hybrid"
RESAMPLE_TECHNIQUE = "smoteenn"

NETWORK = "ResNet50"
LAYERS = 0

SKMODEL_NAME = 'SVM'
SKMODEL_PARAMS = {'C': 5.983876860868068, 'class_weight': 'balanced', 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}

RANDOM_SEARCH = False
RANDOM_SEARCH_ITERATIONS = 50

VALIDATION_TYPE = 'cross_validation'
# CV
SPLITS = 10
# HoldOut
TEST_SIZE = 0.1
REPEATS = 50

MULTIPLECOMPARISON = True
MODELS = {
        "AdaBoost": {'algorithm': 'SAMME', 'learning_rate': 0.6332981268275579, 'n_estimators': 467},
        "DecisionTree": {'class_weight': None, 'criterion': 'gini', 'max_depth': 46, 'min_samples_leaf': 8, 'min_samples_split': 16, 'splitter': 'best'},
        "ExtraTrees": {'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 15, 'n_estimators': 264},
        "GaussianNB": {'var_smoothing': 3.845401188473625e-08},
        "GradientBoosting": {'learning_rate': 0.2479952138102537, 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 69, 'subsample': 0.860829174830409},
        "KNN": {'algorithm': 'kd_tree', 'leaf_size': 35, 'n_neighbors': 43, 'p': 2, 'weights': 'uniform'},
        "LogisticRegression": {'C': 80.61164362700221, 'class_weight': 'balanced', 'max_iter': 269, 'penalty': 'l1', 'solver': 'liblinear'},
        "MLP": {'activation': 'logistic', 'alpha': 0.07229987722668248, 'hidden_layer_sizes': (100,), 'learning_rate': 'invscaling', 'max_iter': 585, 'solver': 'lbfgs'},
        "RandomForest": {'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 25, 'max_features': 'auto', 'min_samples_leaf': 6, 'min_samples_split': 13, 'n_estimators': 394},
        "SVM": {'C': 5.983876860868068, 'class_weight': 'balanced', 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}
    }