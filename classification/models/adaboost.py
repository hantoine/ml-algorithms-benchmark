from sklearn.ensemble import AdaBoostClassifier
from utils import CategoryEncoder
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class AdaBoostModel:
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
        return AdaBoostClassifier(random_state=0, **args)

    hp_space = {
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(10.5), np.log(1000.5), 1)),
        'learning_rate': hp.lognormal('learning_rate', np.log(0.01), np.log(10.0))
    }
