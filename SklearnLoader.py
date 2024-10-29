from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


class SklearnLoader:
    def __init__(self, model_name='RandomForest', model_params=None):
        self.model_name = model_name
        self.model_params = model_params if model_params else {}
        self.model = self._load_model()

    def _load_model(self):
        model_dict = {
            'RandomForest': RandomForestClassifier,
            'SVM': SVC,
            'LogisticRegression': LogisticRegression,
            'GradientBoosting': GradientBoostingClassifier,
            'DecisionTree': DecisionTreeClassifier,
            'KNN': KNeighborsClassifier,
            'AdaBoost': AdaBoostClassifier,
            'ExtraTrees': ExtraTreesClassifier,
            'MLP': MLPClassifier,
            'GaussianNB': GaussianNB
        }

        if self.model_name not in model_dict:
            raise ValueError(f"Model not available, choose from: {list(model_dict.keys())}")

        return model_dict[self.model_name](**self.model_params)

    def get_model(self):
        return self.model
