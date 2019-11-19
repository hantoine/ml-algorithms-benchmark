from .random_forests import RandomForestsModel
from .svm import SVMModel
from .lr import LRModel
from .gaussian_nb import GaussianNBModel

all_models = [
    RandomForestsModel,
    SVMModel,
    LRModel,
    GaussianNBModel
]
