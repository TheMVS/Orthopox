import Config
from KerasLoader import KerasLoader
from Loader import Loader
from ModelEvaluator import ModelEvaluator
from SampleBalancer import SampleBalancer
from SklearnLoader import SklearnLoader
from collections import Counter

def main():
    kerasLoader = KerasLoader()
    model = kerasLoader.load_freeze_and_modify_model(name=Config.NETWORK, num_layers_to_remove=Config.LAYERS)
    model.summary()

    ruta_base = Config.BASE_PATH
    loader = Loader(ruta_base, model)

    X, Y, classes = loader.load_data()

    with open(Config.DATA_SUMMARY_FILE, 'w') as f:
        f.write(f"Total Loaded Images: {len(X)}\n")
        f.write(f"Total Loaded Labels: {len(Y)}\n")
        f.write(f"Class dictionary: {classes}\n")

    if Config.RANDOM_SEARCH:
        from SklearnModelRandomSearch import SklearnModelRandomSearch
        searcher = SklearnModelRandomSearch(X=X, Y=Y, test_size=Config.TEST_SIZE, random_state=Config.SEED,
                                            n_iter=Config.RANDOM_SEARCH_ITERATIONS, cv=Config.SPLITS)
        searcher.run_search_and_save(filename=Config.RANDOMSEARCH_FILE, file_format=Config.RANDOMSEARCH_FORMAT)
    else:
        skLoader = SklearnLoader(model_name=Config.SKMODEL_NAME, model_params=Config.SKMODEL_PARAMS)
        evaluator = ModelEvaluator(skLoader.model, X, Y,
                                   validation_type=Config.VALIDATION_TYPE,
                                   test_size=Config.TEST_SIZE, random_state=Config.SEED, n_splits=Config.SPLITS,
                                   n_repeats=Config.REPEATS)
        evaluator.save_results(loader, results_filename=Config.RESULTS_FILE, stats_filename=Config.STATS_FILE)


if __name__ == "__main__":
    main()
