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
    model = kerasLoader.load_freeze_and_modify_model(name="ResNet50V2", num_layers_to_remove=0) # 23,587,712
    model.summary()

    #from keras.applications import ResNet50
    #resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #resnet50_model.summary()

    print(f"Total Loaded Images: {len(X)}")
    print(f"Total Loaded Labels: {len(Y)}")
    print(f"Class dictionary: {classes}")


if __name__ == "__main__":
    main()
