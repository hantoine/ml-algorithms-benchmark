from sklearn.gaussian_process import GaussianProcessRegressor
from utils import NonTreeBasedModel
from config import RANDOM_STATE
from hyperopt import hp
import numpy as np


class GaussianProcessModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, train_data=None, test=False):
        """
            Using Ridge to make it possible to use L2 regularizations.
            sklearn.linear_model.LinearRegression does not support any regularization.
        """
        return GaussianProcessRegressor(
            normalize_y=True,
            random_state=RANDOM_STATE,
            **args
        )

    hp_space = {
        'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3)),
        'n_restarts_optimizer': hp.quniform('n_restarts_optimizer', 0, 10, 1),
    }
