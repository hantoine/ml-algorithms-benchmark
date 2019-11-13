#%% Get Dataset
import numpy as np
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from hyperopt import Trials, fmin, tpe, space_eval

from classification import datasets as ds
from classification import models
from classification.metrics import compute_loss
from utils import random_state

HP_TUNING_TRIALS = 30
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
        return compute_loss(dataset.metric, confusion_matrix)

    return objective

for dataset in ds.all_datasets:
    print(f'Dataset: {dataset.__name__}')
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
        best_score = -min(trials.losses())
        best_hp = space_eval(model.hp_space, best)
        print(f'Best {dataset.metric}: {best_score}\nWith hyperparams: {best_hp}')
