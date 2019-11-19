from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from utils import CategoryEncoder, random_state
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

class RandomForestsModel:
    @staticmethod
    def prepare_dataset(train_data, test_data, categorical_features):
        X_train, y_train, *other = train_data
        X_test, y_test = test_data
        ce = CategoryEncoder(categorical_features, method='onehot')
        X_train_enc = ce.fit_transform(X_train, y_train)
        X_test_enc = ce.transform(X_test)

        # if Xs are numpy array, ys needs to be too (for k-fold)
        if type(y_train) == pd.Series:
            y_train, y_test = y_train.values, y_test.values

        # Impute missing values if any
        if np.isnan(X_train_enc).any() or np.isnan(X_test_enc).any():
            imp = IterativeImputer(max_iter=10, random_state=random_state)
            X_train_enc = imp.fit_transform(X_train_enc)
            X_test_enc = imp.transform(X_test_enc)

        return (X_train_enc, y_train, *other), (X_test_enc, y_test)

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

