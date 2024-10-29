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
    loader = Loader(ruta_base)

    X, Y, classes = loader.load_data()

    X_aug, Y_aug, classes_aug = loader.data_augmentation(X, Y, prob_flip_horizontal=Config.PROB_FLIP_HORIZONTAL,
                                                         prob_flip_vertical=Config.PROB_FLIP_VERTICAL,
                                                         prob_blur=Config.PROB_BLUR,
                                                         blur_size=Config.BLUR_SIZE)

    X = loader.image_to_model_features(X, model)
    X_aug = loader.image_to_model_features(X_aug, model)


    if Config.DATA_AUGMENTATION:
        X, Y, classes = X_aug, Y_aug, classes_aug


    balancer = SampleBalancer(random_state=Config.SEED)
    X_res, Y_res = balancer.balance(X, Y, method=Config.RESAMPLE_METHOD, technique=Config.RESAMPLE_TECHNIQUE)

    print("Distribución original:")
    print("Y:", Counter(Y))
    print("\nData augmentation:")
    print("Y_res:", Counter(Y_aug))
    print("\nDistribución después de balanceo (SMOTEEEN):")
    print("Y_res:", Counter(Y_res))

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
        evaluator = ModelEvaluator(skLoader.model, X, Y, X_aug, Y_aug, X_res, Y_res,
                                   validation_type=Config.VALIDATION_TYPE,
                                   test_size=Config.TEST_SIZE, random_state=Config.SEED, n_splits=Config.SPLITS,
                                   n_repeats=Config.REPEATS)
        evaluator.save_results(results_filename=Config.RESULTS_FILE, significance_filename=Config.STATS_FILE)


if __name__ == "__main__":
    main()
