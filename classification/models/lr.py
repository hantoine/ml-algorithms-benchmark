from sklearn.linear_model import LogisticRegression 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import CategoryEncoder, random_state
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
import scipy

class LRModel:
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
    def build_estimator(args, test=False):
        return LogisticRegression(
            random_state=random_state,
            solver='saga',
            **args
        )
    
    hp_space = { 
        'penalty': 'elasticnet',
        'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
        'class_weight': hp.pchoice('class_weight', [
            (0.5, None),
            (0.5, 'balanced'),
        ]),
        'multi_class': 'multinomial'
    }

