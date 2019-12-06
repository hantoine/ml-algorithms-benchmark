import json
import time
import warnings
from datetime import timedelta
import math
from os import makedirs
from os.path import join as joinpath

import numpy as np
from hyperopt import Trials, fmin, space_eval, tpe
from hyperopt.mongoexp import MongoTrials
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import shuffle

from config import K_FOLD_K_VALUE, RANDOM_STATE, RESULTS_DIR
from utils import compute_loss, compute_metric
from .timeout import set_timeout, TimeoutError


def tune_all_models_on_all_datasets(task_type, datasets, models, tuning_trials_per_step=5,
                                    max_tuning_time=120, max_trials_without_improvement=150,
                                    tuning_step_max_time=60, mongo_address=None):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    minimum_runtime = max_tuning_time * len(models) * len(datasets)
    print(f'Expected minimum runtime: {timedelta(seconds=minimum_runtime)}')

    for dataset in datasets:
        print(f'Dataset: {dataset.__name__}')
        train, test = dataset.get()
        for model in models:
            print(f'Model: {model.__name__}')
            try:
                train_data, _ = model.prepare_dataset(train, test,
                                                      dataset.categorical_features)

                tune_hyperparams(task_type, dataset, model, train_data,
                                 tuning_trials_per_step, max_tuning_time,
                                 max_trials_without_improvement, tuning_step_max_time,
                                 mongo_address)
            except MemoryError:
                print('Memory requirements for this model with this dataset are too high')


def tune_hyperparams(task_type, dataset, model, train_data, tuning_step_size, max_tuning_time,
                     max_trials_wo_improvement, tuning_step_max_time, mongo_address):
    kfold, train_data = create_kfold(task_type, dataset, train_data)
    objective_fct = create_tuning_objective(dataset, model, train_data, kfold)

    # No tuning for models without hyper-parameters
    is_model_tunable = hasattr(model, 'hp_space')
    if not is_model_tunable:
        loss = objective_fct(None)
        print(f'Resulting {dataset.metric}: {-loss}')
        return

    if tuning_step_max_time > 0:
        make_tuning_step_w_timeout = set_timeout(make_tuning_step, tuning_step_max_time)
    else:
        make_tuning_step_w_timeout = make_tuning_step

    # Tuning loop
    if mongo_address is not None:
        trials = MongoTrials(mongo_address, exp_key=f'{dataset.__name__}-{model.__name__}')
    else:
        trials = Trials()
    start_time = time.time()
    rstate = np.random.RandomState(RANDOM_STATE)
    n_trials_wo_improvement = 0
    time_left = True
    while time_left and n_trials_wo_improvement < max_trials_wo_improvement:
        try:
            make_tuning_step_w_timeout(objective_fct, model.hp_space, trials, rstate, tuning_step_size)
        except TimeoutError:
            pass
        n_trials_wo_improvement = update_n_trials_wo_improvement(trials)
        time_left = (time.time() - start_time) < max_tuning_time
    tuning_time = time.time() - start_time

    process_tuning_result(trials, tuning_time, model, dataset)


def update_n_trials_wo_improvement(trials):
    if len(trials.trials) == 0:
        return 0
    best_trial = min(trials.trials,
                     key=lambda r: r['result']['loss'] if r['result']['status'] == 'ok' else math.inf)
    best_trial_index = sorted(t['tid'] for t in trials.trials).index(best_trial['tid'])

    return len(trials.trials) - best_trial_index


def process_tuning_result(trials, tuning_time, model, dataset):
    n_sucessful_trials = len([None for t in trials.trials if t['result']['status'] == 'ok'])
    if n_sucessful_trials == 0:
        print('No trials finished within allowed time')
        return

    tuning_results_dir = joinpath(RESULTS_DIR, dataset.__name__, model.__name__)

    best_trial = min(trials.trials,
                     key=lambda r: r['result']['loss'] if r['result']['status'] == 'ok' else math.inf)
    best_trial_index = sorted(t['tid'] for t in trials.trials).index(best_trial['tid'])
    best_loss = best_trial['result']['loss']
    best_hp_raw = {k: v[0] if len(v) else None for k, v in best_trial['misc']['vals'].items()}
    best_hp = space_eval(model.hp_space, best_hp_raw)

    best_score = -best_loss if dataset.is_metric_maximized else best_loss

    save_tuning_results(tuning_results_dir, best_hp, best_score, best_trial_index, tuning_time,
                        dataset.is_metric_maximized)

    print(f'Best {dataset.metric}: {best_score:.2f}')
    print(f'With hyperparams: \n{best_hp}')
    print(f'Obtained after {best_trial_index} trials')
    print(f'Total number of sucessful trials: {n_sucessful_trials}')
    print(f'Total tuning time: {tuning_time:.1f}s')

def make_tuning_step(objective_fct, hp_space, trials, rstate, step_size):
    fmin(objective_fct,
         hp_space,
         algo=tpe.suggest,
         max_evals=len(trials.trials) + step_size,
         trials=trials,
         show_progressbar=True,
         rstate=rstate)


def create_tuning_objective(dataset, model, train, kfold):
    def objective(args):
        try:
            estimator = model.build_estimator(args, train)
            metric_values = []
            X, y, *_ = train
            for train_index, val_index in kfold.split(*train):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                estimator.fit(X_train, y_train)
                metric_value = compute_metric(y_val, estimator.predict(X_val), dataset.metric)
                metric_values.append(metric_value)
                if not getattr(dataset, 'needs_k_fold', True):
                    break

            return compute_loss(dataset.metric, metric_values)
        except ValueError:
            """ With some hyper-parameters combinations, a ValueError can be raised during training
                (in particular MLPRegressor)
            """
            return {'status': 'fail'}

    return objective


def create_kfold(task_type, dataset, train_data):
    if task_type == 'classification':
        n_splits = min(K_FOLD_K_VALUE, dataset.get_min_k_fold_k_value(train_data))
        kfold = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    elif task_type == 'regression':
        if getattr(dataset, 'need_grouped_split', False):
            train_data = shuffle(*train_data, random_state=RANDOM_STATE)
            kfold = GroupKFold(n_splits=K_FOLD_K_VALUE)
        else:
            kfold = KFold(n_splits=K_FOLD_K_VALUE, shuffle=True, random_state=RANDOM_STATE)
    return kfold, train_data


def save_tuning_results(tuning_results_dir, hyperparams, score, n_trials, tuning_time,
                        is_metric_maximized):
    makedirs(tuning_results_dir, exist_ok=True)

    try:
        with open(joinpath(tuning_results_dir, 'tuning.json'), 'r', encoding='utf-8') as file:
            prev_results = json.load(file)
            if is_metric_maximized:
                better_results = score > prev_results['score']
            else:
                better_results = score < prev_results['score']
    except FileNotFoundError:
        better_results = True

    if better_results:
        results = {
            'hp': hyperparams,
            'score': score,
            'n_trials': int(n_trials),
            'tuning_time': tuning_time
        }
        with open(joinpath(tuning_results_dir, 'tuning.json'), 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
