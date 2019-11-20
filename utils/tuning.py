import time
import warnings
import numpy as np
from datetime import timedelta
from os.path import dirname, join as joinpath
from os import makedirs

from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
import json

from utils import compute_metric, compute_loss
from config import K_FOLD_K_VALUE, RANDOM_STATE
import classification.datasets as classification_ds
from config import HYPERPARAMS_DIR

def tune_all_models_on_all_datasets(task_type, datasets, models, tuning_trials_per_step=5,
                                    tuning_time=120):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    minimum_runtime = tuning_time * len(models) * len(datasets)
    print(f'Expected minimum runtime: {timedelta(seconds=minimum_runtime)}')

    for dataset in datasets:
        print(f'Dataset: {dataset.__name__}')
        train, test = dataset.get()
        for model in models:
            print(f'Model: {model.__name__}')
            try:
                train_data, test_data = model.prepare_dataset(train, test, dataset.categorical_features)

                best_hp = tune_hyperparams(task_type, dataset, model, train_data,
                                           tuning_trials_per_step, tuning_time)

                save_hyperparams_as_json(best_hp, joinpath(HYPERPARAMS_DIR, dataset.__name__, model.__name__, 'hp.json'))
            except MemoryError:
                print('Memory requirements for this model with this dataset too high')


def tune_hyperparams(task_type, dataset, model, train_data, tuning_step_size, tuning_time):
    kfold, train_data = create_kfold(task_type, dataset, train_data)
    objective_fct = create_tuning_objective(dataset, model, train_data, kfold)

    # For models without hyper-parameters
    is_model_tunable = hasattr(model, 'hp_space')
    if not is_model_tunable:
        loss = objective_fct(None)
        print(f'Resulting {dataset.metric}: {-loss}')
        return {}
    
    trials = Trials()
    rstate = np.random.RandomState(RANDOM_STATE)
    start_time = time.time()
    while time.time() - start_time < tuning_time:
        best = make_tuning_step(objective_fct, model.hp_space, trials, rstate, tuning_step_size)
    tuning_time = time.time() - start_time

    best_score = -min(trials.losses())
    best_hp = space_eval(model.hp_space, best)
    best_trial_index = np.array(trials.losses()).argmin()
    print(f'Best {dataset.metric}: {best_score}')
    print(f'With hyperparams: \n{best_hp}')
    print(f'Obtained after {best_trial_index-1} trials')
    print(f'Total tuning time: {tuning_time:.0f}s')
    return best_hp


def make_tuning_step(objective_fct, hp_space, trials, rstate, step_size):
    return fmin(objective_fct,
                hp_space,
                algo=tpe.suggest,
                max_evals=len(trials.trials) + step_size,
                trials=trials,
                show_progressbar=True,
                rstate=rstate)

def create_tuning_objective(dataset, model, train, kfold):
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

def create_kfold(task_type, dataset, train_data):
    if task_type == 'classification':
        n_splits = min(K_FOLD_K_VALUE, classification_ds.get_min_k_fold_k_value(train_data))
        kfold = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    elif task_type == 'regression':
        if getattr(dataset, 'need_grouped_split', False):
            train_data = shuffle(*train_data, random_state=RANDOM_STATE)
            kfold = GroupKFold(n_splits=K_FOLD_K_VALUE)
        else:
            kfold = KFold(n_splits=K_FOLD_K_VALUE, shuffle=True, random_state=RANDOM_STATE)
    return kfold, train_data

def save_hyperparams_as_json(hyperparams, path):
    makedirs(dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(hyperparams, file, ensure_ascii=False, indent=4)
