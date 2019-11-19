from copy import deepcopy
import warnings
import time
from datetime import timedelta
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from hyperopt import Trials, fmin, tpe, space_eval

from classification import datasets as ds
from classification import models
from utils import random_state, compute_metric, compute_loss

HP_TUNING_STEP_TRIALS = 5
HP_TUNE_MAX_TIME = 120
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
            
        return compute_loss(dataset.metric, metric_values)

    return objective

def tune_hyperparams(trials, dataset, model, train_data, kfold, rstate):
    return fmin(get_objective(dataset, model, train_data, kfold),
                model.hp_space,
                algo=tpe.suggest,
                max_evals=len(trials.trials) + HP_TUNING_STEP_TRIALS,
                trials=trials,
                show_progressbar=True,
                rstate=rstate)

minimum_runtime = HP_TUNE_MAX_TIME * len(models.all_models) * len(ds.all_datasets)
print(f'Expected minimum runtime: {timedelta(seconds=minimum_runtime)}')

for dataset in ds.all_datasets:
    print(f'Dataset: {dataset.__name__}')
    train, test = dataset.get()
    kfold = StratifiedKFold(n_splits=min(K_FOLD_K_VALUE, ds.get_min_k_fold_k_value(train)),
                            shuffle=True, random_state=random_state)
    for model in models.all_models:
        print(f'Model: {model.__name__}')

        # Pre-processings are only applied on the data used by this model
        train_prepared, test_prepared = deepcopy((train, test))
        train_prepared, test_prepared = \
            model.prepare_dataset(train_prepared, test_prepared, dataset.categorical_features)

        # For models without hyper-parameters
        is_model_tunable = hasattr(model, 'hp_space')
        if not is_model_tunable:
            loss = get_objective(dataset, model, train_prepared, kfold)(None)
            print(f'Resulting {dataset.metric}: {-loss}')
            continue

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
