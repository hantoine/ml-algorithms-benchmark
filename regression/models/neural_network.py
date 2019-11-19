from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import CategoryEncoder, random_state
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
import scipy

class NeuralNetworkModel:
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
        return MLPRegressor(
            random_state=random_state,
            max_iter=(1 if test else 300),
            **args
        )

    """
        In order to facilitate hyper-parameter tuning, we choose not to tune betas hyperparameters
        used by the Adam optimizer, the power exponent used by the invscaling learning rate
        schedule and the batch size.

        We choose to use only stochastic gradient-based optimizers because other optimizers would not
        work with some of the big datasets.
    """
    layer_size = scope.int(hp.quniform('layer_size', 10, 100, 5))

    hp_space = { 
        'hidden_layer_sizes': hp.choice('n_layers', [
                [layer_size],
                [layer_size]*2,
                [layer_size]*3,
            ]),
        'early_stopping': True,
        'activation': hp.pchoice('activation', [
            (0.25, 'logistic'),
            (0.25, 'tanh'),
            (0.5, 'relu')
            ]),
        'solver': hp.choice('solver', ['sgd', 'adam']),
        'alpha': hp.loguniform('alpha', np.log(1e-7), np.log(1e-2)),
        'batch_size': 128,
        'learning_rate': hp.pchoice('learning_rate_schedule', [
                (0.2, 'constant'),
                (0.3, 'invscaling'),
                (0.5, 'adaptive')
            ]),
        'learning_rate_init': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
        'momentum': hp.uniform('momentum', 0.87, 0.99),
        'nesterovs_momentum': hp.pchoice('nesterovs_momentum', [
                (0.7, True),
                (0.3, False)
            ]),
    }

