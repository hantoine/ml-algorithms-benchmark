from .random_forests import RandomForestsModel
from .svm import SVMModel
from .linear_regression import LinearRegressionModel
from .gaussian_process import GaussianProcessModel
from .neural_network import NeuralNetworkModel
from .decision_tree import DecisionTreeModel
from .adaboost import AdaBoostModel
from .k_nearest_neighbors import KNearestNeighborsModel
from .gradient_boosting import GradientBoostingModel

all_models = [
    RandomForestsModel,
    SVMModel,
    LinearRegressionModel,
    GaussianProcessModel,
    DecisionTreeModel,
    AdaBoostModel,
    KNearestNeighborsModel,
    NeuralNetworkModel,
    GradientBoostingModel
]
