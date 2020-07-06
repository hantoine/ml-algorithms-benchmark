from glob import glob
from config import RESULTS_DIR
import json
import re
import pandas as pd
from classification import datasets as clf_ds
from regression import datasets as reg_ds
from utils.critical_difference_diagram import draw_cd_diagram


def get_results_table():
    result_files = glob('results/*/*/evaluation.json')

    pattern = re.compile(RESULTS_DIR + r'/([a-zA-Z0-9]+)/([a-zA-Z0-9]+)/evaluation.json')
    result_table = []

    for result_file in result_files:
        dataset, model = pattern.match(result_file).groups()
        with open(result_file, 'r') as file:
            results = json.load(file)
        results['hp'] = ','.join([f"{k}={v}" if type(v) in [str, list, type(None)] else f"{k}={v:.2f}"
                                  for k, v in results['hp'].items()])
        results.update({'dataset': dataset, 'model': model})
        result_table.append(results)

    result_table = pd.DataFrame(result_table)
    result_table = result_table[['dataset', 'model', 'score', 'val_score', 'train_time',
                                 'evaluation_time', 'tuning_n_trials', 'hp']]
    return result_table


def get_model_ranking(results):
    results['model_rank'] = results.groupby('dataset')['score'].rank(ascending=False)
    return results.groupby('model')['model_rank'].mean().sort_values().rename('rank')


def split_results_by_task(results):
    def dataset_to_task(dataset):
        if hasattr(clf_ds, dataset):
            return 'classification'
        elif hasattr(reg_ds, dataset):
            return 'regression'
        else:
            return 'classifier_interpretability'

    def dataset_to_metric(ds):
        ds_class = (getattr(reg_ds, ds, None) or getattr(clf_ds, ds, None))
        if ds_class is None:  # dataset for classifier interpretability
            return None
        return ds_class.metric

    results['task'] = results['dataset'].apply(dataset_to_task)
    results['metric'] = results['dataset'].apply(dataset_to_metric)

    # Remove "Dataset" and "Model" suffixes
    results.loc[:, 'dataset'] = results['dataset'].str[:-7]
    results.loc[:, 'model'] = results['model'].str[:-5]

    clf_results = results[results['task'] == 'classification'].copy()
    reg_results = results[results['task'] == 'regression'].copy()
    clf_interpretability_results = results[results['task'] == 'classifier_interpretability'].copy()

    return clf_results, reg_results, clf_interpretability_results


def print_results(format='markdown'):
    results = get_results_table()
    clf_results, reg_results, _ = split_results_by_task(results)

    print('Classification models global ranking:')
    print(get_model_ranking(clf_results))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification.png', clf_results)

    print('\nClassification models small dataset (<10k examples) ranking')
    big_datasets = ('DefaultCreditCard', 'Adult')
    clf_result_small_ds = clf_results[~clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_small_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification-small-ds.png', clf_result_small_ds)

    print('\nClassification models big dataset (>10k examples) ranking')
    big_datasets = ('DefaultCreditCard', 'Adult')
    clf_result_big_ds = clf_results[clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_big_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification-big-ds.png', clf_result_big_ds)

    print('\nRegression models global ranking:')
    print(get_model_ranking(reg_results))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression.png', reg_results)

    print('\nRegression models small dataset (<5k examples) ranking')
    big_datasets = ('MerckMolecularActivity', 'BikeSharing', 'SGEMMGPUKernelPerformances')
    reg_result_small_ds = reg_results[~reg_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(reg_result_small_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression-small-ds.png', reg_result_small_ds)

    print('\nRegression models big dataset (>5k examples) ranking')
    big_datasets = ('MerckMolecularActivity', 'BikeSharing', 'SGEMMGPUKernelPerformances')
    reg_result_big_ds = reg_results[reg_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(reg_result_big_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression-big-ds.png', reg_result_big_ds)

    print_raw_results('Classification', clf_results, format)
    print_raw_results('Regression', reg_results, format)

    print_rankings('Classification', clf_results, format)
    print_rankings('Regression', reg_results, format)


def print_rankings(task, results, format='default', use_short_alg_names=True):
    print(f'{task} ranking table')
    results['model_rank'] = results.groupby('dataset')['score'].rank(ascending=False)
    results = results[['dataset', 'model', 'model_rank']]
    table = pd.pivot_table(results, values='model_rank', index='model', columns=['dataset'])

    if task == 'Regression':
        big_ds = ['SGEMMGPUKernelPerformances', 'MerckMolecularActivity', 'BikeSharing']
    else:
        big_ds = ['DefaultCreditCard', 'Adult']
    small_ds = list(set(table.columns) - set(big_ds))

    table['Average'] = table.T.mean()
    table['Average Big Dataset'] = table[big_ds].T.mean()
    table['Average Small Dataset'] = table[small_ds].T.mean()

    if use_short_alg_names:
        def keep_capital_letters_only(s):
            return ''.join(c for c in s if c.isupper())
        table.index = table.index.to_series().apply(keep_capital_letters_only)

    table = table.round(1)
    for col in table.columns[:-3]:
        table.loc[:, col] = table[col].astype(str).str[:-2]
    table = table.T
    table = table.replace('n', 'F')
    if format == 'latex':
        print(table.to_latex())
    elif format == 'markdown':
        print(table.to_markdown())
    elif format == 'default':
        print(table)
    else:
        raise ValueError('Unknown format')


def print_raw_results(task, results, format='default', use_short_alg_names=True):
    results = results[['dataset', 'model', 'score', 'metric']]

    results = results.replace('accuracy', 'acc')
    table = pd.pivot_table(results, values='score', index='model', columns=['dataset', 'metric'])

    if use_short_alg_names:
        def keep_capital_letters_only(s):
            return ''.join(c for c in s if c.isupper())
        table.index = table.index.to_series().apply(keep_capital_letters_only)

    table = table.round(2).T
    if format == 'latex':
        print(table.to_latex())
    elif format == 'markdown':
        print(table.to_markdown())
    elif format == 'default':
        print(table)
    else:
        raise ValueError('Unknown format')


def print_all_datasets_results(datasets, results, format='default'):
    dataset_names = results.dataset.unique()
    for dataset_name in dataset_names:
        dataset = getattr(datasets, dataset_name + 'Dataset')
        table = (results[results.dataset == dataset_name]
                 .set_index('model')
                 [['score', 'train_time']]
                 .sort_values('score', ascending=not dataset.is_metric_maximized)
                 .rename(columns={'score': dataset.metric}))
        table.index = table.index.rename('')
        table[dataset.metric] = abs(table[dataset.metric])
        if dataset.metric in ('accuracy', 'f1'):
            table[dataset.metric] = (table[dataset.metric] * 100)
        table = table.round(1)

        if format == 'latex':
            print("\\begin{figure}\\caption{Results on " + dataset_name[:-7] + " dataset}")
            print(table.to_latex(column_format='|lrr|'))
            print('\\end{figure}')
        elif format == 'markdown':
            print(table.to_markdown())
        elif format == 'default':
            print(f'Results on {dataset_name[:-7]} dataset')
            print(table)
        else:
            raise ValueError('Unknown format')
