import Config
from KerasLoader import KerasLoader
from Loader import Loader


def main():
    ruta_base = Config.BASE_PATH
    loader = Loader(ruta_base)

    X, Y, classes = loader.load_data()

    if Config.DATA_AUGMENTATION:
        X, Y, classes = loader.data_augmentation(X, Y, prob_flip_horizontal=Config.PROB_FLIP_HORIZONTAL,
                                                 prob_flip_vertical=Config.PROB_FLIP_VERTICAL,
                                                 prob_blur=Config.PROB_BLUR,
                                                 blur_size=Config.BLUR_SIZE)

    kerasLoader = KerasLoader()
    model = kerasLoader.load_freeze_and_modify_model(name=Config.NETWORK, num_layers_to_remove=Config.LAYERS)
    model.summary()

    new_X = loader.image_to_model_features(X, model)

    if Config.RANDOM_SEARCH:
        from SklearnModelRandomSearch import SklearnModelRandomSearch
        searcher = SklearnModelRandomSearch(X=new_X, Y=Y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
        searcher.run_search_and_save()

    print(f"Total Loaded Images: {len(X)}")
    print(f"Total Loaded Labels: {len(Y)}")
    print(f"Class dictionary: {classes}")


if __name__ == "__main__":
    main()
