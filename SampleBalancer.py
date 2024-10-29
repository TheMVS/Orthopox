from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

class SampleBalancer:
    def __init__(self, random_state=42):
        self.random_state = random_state

        self.undersampling_methods = {
            "random": RandomUnderSampler(random_state=self.random_state),
            "nearmiss": NearMiss(),
            "cluster_centroids": ClusterCentroids(random_state=self.random_state)
        }

        self.oversampling_methods = {
            "random": RandomOverSampler(random_state=self.random_state),
            "smote": SMOTE(random_state=self.random_state),
            "adasyn": ADASYN(random_state=self.random_state)
        }

        self.hybrid_methods = {
            "smoteenn": SMOTEENN(random_state=self.random_state),
            "smotetomek": SMOTETomek(random_state=self.random_state)
        }

    def balance(self, X, y, method="undersample", technique="random"):
        if method == "undersample":
            sampler = self.undersampling_methods.get(technique)
        elif method == "oversample":
            sampler = self.oversampling_methods.get(technique)
        elif method == "hybrid":
            sampler = self.hybrid_methods.get(technique)
        else:
            raise ValueError("Not valid method, use: 'undersample', 'oversample', o 'hybrid'.")

        if sampler is None:
            raise ValueError(f"Technique '{technique}' not valid for method '{method}'.")

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
