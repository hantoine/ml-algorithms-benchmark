import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import shuffle
from hyperopt import Trials, fmin, tpe, space_eval

from regression import datasets as ds
from regression import models
from regression.metrics import compute_metric, aggregate_metrics
from utils import random_state

HP_TUNING_TRIALS = 1
K_FOLD_K_VALUE = 7

def get_objective(dataset, model, train, kfold):
    def objective(args):
        estimator = model.build_estimator(args)
        metric_values = []
        X, y, *_ = train
        for train_index, val_index in kfold.split(*train):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            estimator.fit(X_train, y_train)
            metric_value = compute_metric(y_val, estimator.predict(X_val), dataset.metric)
            metric_values.append(metric_value)
            
        score = aggregate_metrics(metric_values, dataset.metric)
        return -score

    return objective

for dataset in ds.all_datasets:
    print(f'Dataset: {dataset.__name__}')
    train, test = dataset.get()
    if getattr(dataset, 'need_grouped_split', False):
        train = shuffle(*train, random_state=random_state)
        kfold = GroupKFold(n_splits=K_FOLD_K_VALUE)
    else:
        kfold = KFold(n_splits=K_FOLD_K_VALUE, shuffle=True, random_state=random_state)
    for model in [models.RandomForestsModel]:
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        trials = Trials()
        best = fmin(get_objective(dataset, model, train, kfold),
                    model.hp_space,
                    algo=tpe.suggest,
                    max_evals=HP_TUNING_TRIALS,
                    trials=trials,
                    show_progressbar=True,
                    rstate=np.random.RandomState(random_state))
        best_score = -min(trials.losses())
        best_hp = space_eval(model.hp_space, best)
        print(f'Best {dataset.metric}: {best_score}\nWith hyperparams: {best_hp}')
