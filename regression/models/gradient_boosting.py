from sklearn.ensemble import GradientBoostingRegressor
from utils import TreeBasedModel
from config import RANDOM_STATE
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class GradientBoostingModel(TreeBasedModel):
    @staticmethod
    def build_estimator(args):
        return GradientBoostingRegressor(random_state=RANDOM_STATE, presort=True, **args)

    loss_alpha = hp.choice('loss_alpha', [
        ('ls', 0.9),
        ('lad', 0.9),
        ('huber', hp.uniform('gbr' + '.alpha', 0.85, 0.95)),
        ('quantile', 0.5)
    ])

    hp_space = {
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(10.5), np.log(1000.5), 1)),
        'learning_rate': hp.lognormal('learning_rate', np.log(0.01), np.log(10.0)),
        'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae']),
        'max_depth': hp.pchoice('max_depth', [
            (0.2, 2),
            (0.5, 3),
            (0.2, 4),
            (0.1, 5)
        ]),
        'min_samples_leaf': hp.choice('min_samples_leaf_enabled', [
            1,  # most common choice.
            scope.int(hp.qloguniform('min_samples_leaf' + '.gt1', np.log(1.5), np.log(50.5), 1))
        ]),
        'subsample': hp.pchoice('subsample_enabled', [
            (0.2, 1.0),  # default choice.
            (0.8, hp.uniform('subsample' + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
        ]),
        'max_features': hp.pchoice('max_features_str', [
            (0.1, 'sqrt'),  # most common choice.
            (0.2, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform('max_features_str_frac', 0., 1.))
        ]),
        'loss': loss_alpha[0],
        'alpha': loss_alpha[1]
    }
