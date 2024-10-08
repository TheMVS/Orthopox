from Loader import Loader


def main():
    ruta_base = "./Data"
    loader = Loader(ruta_base)

    X, Y, classes = loader.load_data()

    X, Y, classes = loader.data_augmentation(X, Y)

    print(f"Loaded Images: {len(X)}")
    print(f"Loaded Labels: {len(Y)}")
    print(f"Class dicctionay: {classes}")

if __name__ == "__main__":
    main()