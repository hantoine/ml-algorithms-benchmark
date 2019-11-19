from .random_forests import RandomForestsModel
from .svm import SVMModel
from .linear_regression import LinearRegressionModel
from .neural_network import NeuralNetworkModel
from .gaussian_process import GaussianProcessModel

all_models = [
    RandomForestsModel,
    SVMModel,
    LinearRegressionModel,
    NeuralNetworkModel,
    GaussianProcessModel
]
