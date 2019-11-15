from sklearn.tree import DecisionTreeClassifier
from utils import CategoryEncoder
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class DecisionTreeModel:
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
        return DecisionTreeClassifier(random_state=0, **args)

    hp_space = {
        'max_depth': hp.pchoice('max_depth_enabled', [
            (0.7, None),
            (0.3, scope.int(hp.qlognormal('max_depth', np.log(30), 0.5, 3)))]),
        'splitter': hp.choice('splitter_str', ['best', 'random']),
        'max_features': hp.pchoice('max_features_str', [
            (0.2, 'sqrt'),  # most common choice.
            (0.1, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform('max_features_str_frac', 0., 1.))
        ]),
        'min_samples_split': scope.int(hp.quniform(
            'min_samples_split_str',
            2, 10, 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf_enabled', [
            1,
            scope.int(hp.qloguniform('min_samples_leaf', np.log(1.5), np.log(50.5), 1))
        ]),
    }