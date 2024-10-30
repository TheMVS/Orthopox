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

SKMODEL_NAME = 'LogisticRegression'
SKMODEL_PARAMS = {'solver': 'saga', 'C': 10}

RANDOM_SEARCH = False
RANDOM_SEARCH_ITERATIONS = 50

VALIDATION_TYPE = 'cross_validation'
# CV
SPLITS = 5
# HoldOut
TEST_SIZE = 0.1
REPEATS = 50