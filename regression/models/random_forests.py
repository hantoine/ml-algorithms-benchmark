from sklearn.ensemble import RandomForestRegressor
from utils import random_state, TreeBasedModel
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class RandomForestsModel(TreeBasedModel):
    @staticmethod
    def build_estimator(args):
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **args)

    hp_space = {
        'max_depth': hp.pchoice('max_depth_enabled', [
            (0.7, None),
            (0.3, scope.int(hp.qlognormal('max_depth', np.log(30), 0.5, 3)))]),
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300), 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf_enabled', [
            1,
            scope.int(hp.qloguniform('min_samples_leaf', np.log(1.5), np.log(50.5), 1))
        ]),
        'max_features': hp.pchoice('max_features_str', [
            (0.1, 'sqrt'),  # most common choice.
            (0.2, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform('max_features_str_frac', 0., 1.))
        ])
    }
