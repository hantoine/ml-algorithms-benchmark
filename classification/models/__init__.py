from .random_forests import RandomForestsModel
from .svm import SVMModel
from .lr import LRModel
from .gaussian_nb import GaussianNBModel
from .decision_tree import DecisionTreeModel
from .adaboost import AdaBoostModel
from .k_nearest_neighbours import KNearestNeighborsModel
from .neural_network import NeuralNetworkModel

all_models = [
    RandomForestsModel,
    SVMModel,
    LRModel,
    GaussianNBModel,
    DecisionTreeModel,
    AdaBoostModel,
    KNearestNeighborsModel,
    NeuralNetworkModel
]
