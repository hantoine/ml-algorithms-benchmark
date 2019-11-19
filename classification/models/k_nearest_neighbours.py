from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import CategoryEncoder, random_state
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
import scipy


class KNearestNeighborsModel:
    @staticmethod
    def prepare_dataset(train_data, test_data, categorical_features):
        X_train, y_train = train_data
        X_test, y_test = test_data
        ce = CategoryEncoder(categorical_features, method='onehot')
        X_train_enc = ce.fit_transform(X_train, y_train)
        X_test_enc = ce.transform(X_test)

        # if Xs are numpy array, ys needs to be too (for k-fold)
        if type(y_train) == pd.Series:
            y_train, y_test = y_train.values, y_test.values

        # StandardScaler does not support sparse matrix
        if type(X_train_enc) == scipy.sparse.csr.csr_matrix:
            X_train_enc = X_train_enc.toarray()
            X_test_enc = X_test_enc.toarray()
        scaler = StandardScaler()
        X_train_enc = scaler.fit_transform(X_train_enc)
        X_test_enc = scaler.transform(X_test_enc)

        # Impute missing values if any
        if np.isnan(X_train_enc).any() or np.isnan(X_test_enc).any():
            imp = IterativeImputer(max_iter=10, random_state=random_state)
            X_train_enc = imp.fit_transform(X_train_enc)
            X_test_enc = imp.transform(X_test_enc)

        return (X_train_enc, y_train), (X_test_enc, y_test)

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
        # 'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'metric': metric_hp[0],
        'p': metric_hp[1]
    }
