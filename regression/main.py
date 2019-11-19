from copy import deepcopy
import warnings
import time
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
from hyperopt import Trials, fmin, tpe, space_eval

from regression import datasets as ds
from regression import models
from regression.metrics import compute_metric, aggregate_metrics
from utils import random_state

HP_TUNING_STEP_TRIALS = 1
HP_TUNE_MAX_TIME = 1
K_FOLD_K_VALUE = 7

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def tune_hyperparams(trials, dataset, model, train_data, kfold, rstate):
    return fmin(get_objective(dataset, model, train_data, kfold),
                model.hp_space,
                algo=tpe.suggest,
                max_evals=len(trials.trials) + HP_TUNING_STEP_TRIALS,
                trials=trials,
                show_progressbar=True,
                rstate=rstate)

for dataset in ds.all_datasets:
    print(f'Dataset: {dataset.__name__}')
    train, test = dataset.get()
    if getattr(dataset, 'need_grouped_split', False):
        train = shuffle(*train, random_state=random_state)
        kfold = GroupKFold(n_splits=K_FOLD_K_VALUE)
    else:
        kfold = KFold(n_splits=K_FOLD_K_VALUE, shuffle=True, random_state=random_state)
    for model in models.all_models:
        print(f'Model: {model.__name__}')
        try:
            # Pre-processings are only applied on the data used by this model
            train_prepared, test_prepared = deepcopy((train, test))
            train_prepared, test_prepared = \
                model.prepare_dataset(train_prepared, test_prepared, dataset.categorical_features)

            trials = Trials()
            rstate = np.random.RandomState(random_state)
            start_time = time.time()
            while time.time() - start_time < HP_TUNE_MAX_TIME:
                best = tune_hyperparams(trials, dataset, model, train_prepared, kfold, rstate)
            tuning_time = time.time() - start_time

            best_score = -min(trials.losses())
            best_hp = space_eval(model.hp_space, best)
            best_trial_index = np.array(trials.losses()).argmin()
            print(f'Best {dataset.metric}: {best_score}')
            print(f'With hyperparams: \n{best_hp}')
            print(f'Obtained after {best_trial_index-1} trials')
            print(f'Total tuning time: {tuning_time:.0f}s')
        except MemoryError:
            print('Memory requirements for this model with this dataset')
