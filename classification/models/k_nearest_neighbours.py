from sklearn.neighbors import KNeighborsClassifier
from utils import NonTreeBasedModel
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class KNearestNeighborsModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args):
        return KNeighborsClassifier(n_jobs=-1, **args)

    metric_hp = hp.pchoice('metric', [
        (0.55, ('euclidean', 2)),
        (0.15, ('manhattan', 1)),
        (0.15, ('chebyshev', 0)),
        (0.15, ('minkowski', hp.quniform('metric_minkowski', 2.5, 5.5, 1))),
    ])

    hp_space = {
        'n_neighbors': scope.int(hp.qloguniform('n_neighbors', np.log(0.5), np.log(50.5), 1)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'metric': metric_hp[0],
        'p': metric_hp[1]
    }
