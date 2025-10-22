# Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models

**Authors:** Alejandro Puente-Castro, Enrique Fernandez-Blanco, Daniel Rivero, Andres Molares-Ulloa

**Author Contributions:**

* **Alejandro Puente-Castro**: Conceptualization, Methodology, Software, Validation, Formal Analysis, Resources, Data Curation, Writing – Original Draft Preparation, Visualization, Investigation
* **Andres Molares-Ulloa**: Reviewing and Editing
* **Enrique Fernandez-Blanco**: Writing – Reviewing and Editing, Project Administration, Funding Acquisition
* **Daniel Rivero**: Reviewing and Editing

This project implements a **complete image classification pipeline** using Keras and scikit-learn. It provides functionality to load images, perform data augmentation, extract features using pretrained neural networks, train machine learning models, evaluate performance, and conduct statistical significance tests between original, augmented, and balanced datasets.

## Dataset

The dataset used in this project is obtained from Kaggle:

[Monkeypox Skin Image Dataset](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset)

It contains skin lesion images of patients with Monkeypox and other conditions, organized into the following classes:

* `Chickenpox`
* `Measles`
* `Monkeypox`
* `Normal`

## Project Structure

```
.
├─ Config.py                 # Global project configurations (paths, models, hyperparameters, data augmentation, balancing, PCA, etc.)
├─ KerasLoader.py            # Loads pretrained Keras models (VGG16, ResNet50, InceptionV3, MobileNetV2, ResNet50V2), freezes layers, removes last layers for feature extraction
├─ Loader.py                 # Loads images, applies optional augmentations, and converts them to features using a pretrained model
├─ ModelEvaluator.py         # Evaluates a single scikit-learn model using holdout, cross-validation, or leave-one-out, saves per-class metrics and significance tests
├─ MultiModelEvaluator.py    # Evaluates multiple scikit-learn models simultaneously, supports PCA, and all validation schemes
├─ SampleBalancer.py         # Balances datasets using methods such as SMOTE, ENN, or hybrid techniques
├─ SklearnLoader.py          # Initializes scikit-learn models with specified hyperparameters
└─ README.md                 # Project documentation
```

## Required Libraries

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn scipy opencv-python tensorflow
```

Optionally, for GPU support in TensorFlow:

```bash
pip install tensorflow-gpu
```

## Configuration (Config.py)

Some important configuration options:

```python
BASE_PATH = "./data"                 # Folder containing dataset
IMAGE_SIZE = (224, 224, 3)           # Image dimensions
DATA_AUGMENTATION = False             # Enable or disable data augmentation
PROB_FLIP_HORIZONTAL = 0.0
PROB_FLIP_VERTICAL = 0.5
PROB_BLUR = 0.5
BLUR_SIZE = 5
RESAMPLE_METHOD = "hybrid"           # Resampling method
RESAMPLE_TECHNIQUE = "smoteenn"      # Resampling technique
NETWORK = "ResNet50"                 # Base Keras network for feature extraction
SKMODEL_NAME = 'SVM'                 # Default scikit-learn model
RANDOM_SEARCH = False
VALIDATION_TYPE = 'cross_validation' # Validation type: holdout, cross_validation, leave_one_out
SPLITS = 10
REPEATS = 50
ATRIBUTE_REDUCTION = 'pca'           # Dimensionality reduction method ('pca')
PCA_COMPONENTS = 1000
BATCH_NORMALIZATION = True
```

## Usage

### 1. Load Images and Classes

```python
from Loader import Loader
from KerasLoader import KerasLoader
import Config

keras_loader = KerasLoader(model_name=Config.NETWORK, input_shape=Config.IMAGE_SIZE)
base_model = keras_loader.load_freeze_and_modify_model(Config.NETWORK, Config.LAYERS)

loader = Loader(base_path=Config.BASE_PATH, model=base_model, size=Config.IMAGE_SIZE[:2])
X, Y, classes = loader.load_data()
```

### 2. Evaluate a Single Model

```python
from sklearn.svm import SVC
from ModelEvaluator import ModelEvaluator
import Config

model = SVC(**Config.SKMODEL_PARAMS)
evaluator = ModelEvaluator(model, X, Y, validation_type=Config.VALIDATION_TYPE,
                           test_size=Config.TEST_SIZE, n_splits=Config.SPLITS, n_repeats=Config.REPEATS)
evaluator.save_results(loader)
```

### 3. Evaluate Multiple Models Simultaneously

```python
from MultiModelEvaluator import MultiModelEvaluator
import Config

multi_evaluator = MultiModelEvaluator(Config.MODELS, X, Y,
                                      validation_type=Config.VALIDATION_TYPE,
                                      test_size=Config.TEST_SIZE, n_splits=Config.SPLITS, n_repeats=Config.REPEATS)
multi_evaluator.save_results(loader)
```

## Calculated Metrics

The evaluation includes:

* Per-class accuracy
* Per-class precision
* Per-class recall
* Per-class F1-score
* Cohen’s Kappa
* Confusion matrix per class
* Statistical significance tests between original, augmented, and resampled datasets

## Dataset Organization

The dataset should be organized into subfolders by class:

```
data/
 ├─ Chickenpox/
 │   ├─ img1.jpg
 │   ├─ img2.jpg
 ├─ Measles/
 │   ├─ img1.jpg
 │   ├─ img2.jpg
 ├─ Monkeypox/
 │   ├─ img1.jpg
 │   ├─ img2.jpg
 └─ Normal/
     ├─ img1.jpg
     ├─ img2.jpg
```

* `MultiModelEvaluator` automatically applies PCA if enabled in `Config.py`.
* All evaluation results are saved as CSV files, including mean and standard deviation for each metric.
