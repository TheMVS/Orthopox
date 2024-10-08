from keras import Model
from keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2

class KerasLoader:
    def __init__(self, model_name="ResNet50", input_shape=(224, 224, 3)):
        self.model_name = model_name
        self.input_shape = input_shape
        self.base_model = None
        self.model_dict = {
            'VGG16': VGG16,
            'ResNet50': ResNet50,
            'InceptionV3': InceptionV3,
            'MobileNetV2': MobileNetV2
        }

    def load_model(self, name):
        self.model_name = name
        if self.model_name not in self.model_dict:
            raise ValueError(f"Model {self.model_name} is not available. Choose from: {list(self.model_dict.keys())}")
        self.base_model = self.model_dict[self.model_name](weights='imagenet', include_top=False, input_shape=self.input_shape)
        return self.base_model

    def freeze_layers(self):
        if self.base_model is None:
            raise ValueError("You must load a model first using `load_model()`.")
        for layer in self.base_model.layers:
            layer.trainable = False

    def remove_last_layers(self, num_layers_to_remove):
        if self.base_model is None:
            raise ValueError("You must load a model first using `load_model()`.")
        if num_layers_to_remove == 0:
            return self.base_model
        if num_layers_to_remove > len(self.base_model.layers):
            raise ValueError(f"Cannot remove {num_layers_to_remove} layers. The model has only {len(self.base_model.layers)} layers.")
        model = self.base_model.layers[len(self.base_model.layers) - num_layers_to_remove -1].output
        model_cut = Model(self.base_model.input, model)
        return model_cut

    def load_freeze_and_modify_model(self, name, num_layers_to_remove=0):
        self.load_model(name)
        self.freeze_layers()
        modified_model = self.remove_last_layers(num_layers_to_remove)
        return modified_model