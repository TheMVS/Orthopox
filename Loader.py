import os
import cv2
import numpy as np


class Loader:
    def __init__(self, base_path, size=(224, 224)):
        self.base_path = base_path
        self.size = size
        self.X = []
        self.Y = []
        self.classes = {}

    def load_data(self):
        class_id = 0

        # Recorremos las subcarpetas dentro de la carpeta principal
        for folder_name in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder_name)

            if os.path.isdir(folder_path):
                self.classes[class_id] = folder_name
                image_count = 0

                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)

                    try:
                        image = cv2.imread(image_path)
                        if image is not None:

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            resize_image = cv2.resize(image_rgb, self.size)
                            self.X.append(resize_image)
                            self.Y.append(class_id)
                            image_count += 1
                        else:
                            print(f"Could not open: {image_path}")
                    except Exception as e:
                        print(f"Could not open {image_path}: {e}")

                # Imprimir cuántas imágenes se encontraron para cada clase
                print(f"Class '{folder_name}' (ID: {class_id}): {image_count} images")
                class_id += 1

        # Convertimos las listas a arrays de NumPy para su uso en modelos de deep learning
        return np.array(self.X), np.array(self.Y), self.classes
