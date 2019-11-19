from sklearn.linear_model import Ridge
from utils import random_state, NonTreeBasedModel
from hyperopt import hp
import numpy as np


class LinearRegressionModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, test=False):
        """
            Using Ridge to make it possible to use L2 regularizations.
            sklearn.linear_model.LinearRegression does not support any regularization.
        """
        return Ridge(
            random_state=random_state,
            # Prevent very long training time for some hyper-parameters
            max_iter=100 if test else 3000,
            **args
        )

    hp_space = {'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3))}
