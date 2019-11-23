from .random_forests import RandomForestsModel
from .svm import SVMModel
from .lr import LRModel
from .gaussian_nb import GaussianNBModel
from .decision_tree import DecisionTreeModel
from .adaboost import AdaBoostModel
from .k_nearest_neighbours import KNearestNeighborsModel
from .basic_neural_network import BasicNeuralNetworkModel
from .gradient_boosting import GradientBoostingModel
from .advanced_neural_network import AdvancedNeuralNetworkModel

all_models = [
    RandomForestsModel,
    SVMModel,
    LRModel,
    GaussianNBModel,
    DecisionTreeModel,
    AdaBoostModel,
    KNearestNeighborsModel,
    BasicNeuralNetworkModel,
    GradientBoostingModel,
    AdvancedNeuralNetworkModel
]
