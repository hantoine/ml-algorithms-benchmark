from sklearn.neighbors import KNeighborsClassifier
from utils import CategoryEncoder
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class RandomForestsModel:
    @staticmethod
    def prepare_dataset(train_data, test_data, categorical_features):
        X_train, y_train = train_data
        X_test, y_test = test_data
        ce = CategoryEncoder(categorical_features, method='sorted_ordinal')
        X_train_enc = ce.fit_transform(X_train, y_train)
        X_test_enc = ce.transform(X_test)
        return X_train_enc, y_train, X_test_enc, y_test

    @staticmethod
    def build_estimator(args):
        return KNeighborsClassifier(random_state=0, n_jobs=-1, **args)

    hp_space = {
        'n_neighbours': scope.int(hp.qloguniform('n_neighbours', np.log(0.5), np.log(50.5), 1)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'metric': hp.pchoice('metric', [
            (0.55, ('euclidean', 2)),
            (0.15, ('manhattan', 1)),
            (0.15, ('chebyshev', 0)),
            (0.15, ('minkowski', hp.quniform('metric_minkowski', 2.5, 5.5, 1))),
        ])
    }
