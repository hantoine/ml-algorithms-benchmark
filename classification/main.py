#%% Get Dataset
import numpy as np
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from hyperopt import Trials, fmin, tpe, space_eval

from classification import datasets as ds
from classification import models
from utils import random_state

HP_TUNING_TRIALS = 1
K_FOLD_K_VALUE = 7

def get_objective(dataset, model, X, y, kfold):
    def objective(args):
        estimator = model.build_estimator(args)
        confusion_matrices = []
        for train_index, val_index in kfold.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            estimator.fit(X_train, y_train)
            conf_matrix = compute_confusion_matrix(y_val, estimator.predict(X_val))
            confusion_matrices.append(conf_matrix)
            
            confusion_matrix = np.array(confusion_matrices).sum(axis=0)

            # Deal with non-binary classification case
            is_binary_classification = confusion_matrix.shape == (2, 2)
            if is_binary_classification:
                tn, fp, fn, tp = confusion_matrix.ravel()
            elif dataset.metric == 'accuracy':
                tp = np.sum(np.diag(confusion_matrix))
                fn = np.sum(confusion_matrix) - tp
                tn, fp = 0, 0
            else:
                raise NotImplementedError

            if dataset.metric == 'f1':
                # Averaging of f1 across folds as suggested by Forman & Scholz
                # https://www.hpl.hp.com/techreports/2009/HPL-2009-359.pdf
                score = (2*tp) / (2*tp + fp + fn)
            elif dataset.metric == 'accuracy':
                score = (tp + tn) / (tn + fp + fn + tp)
            else:
                raise ValueError(f'Metric for dataset f{type(dataset).__name__}'
                                 + ' not implemented (f{dataset.metric})')
            return -score
    return objective

for dataset in ds.all_datasets:
    train, test = dataset.get()
    kfold = StratifiedKFold(n_splits=min(K_FOLD_K_VALUE, ds.get_min_k_fold_k_value(train)),
                            shuffle=True, random_state=random_state)
    for model in [models.RandomForestsModel]:
        X, y, X_test, y_test = \
            model.prepare_dataset(train, test, dataset.categorical_features)
        trials = Trials()
        best = fmin(get_objective(dataset, model, X, y, kfold),
                    model.hp_space,
                    algo=tpe.suggest,
                    max_evals=HP_TUNING_TRIALS,
                    trials=trials,
                    show_progressbar=True,
                    rstate=np.random.RandomState(random_state))
        print(best)
