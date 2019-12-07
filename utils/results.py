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

    for result_file in result_files: # For here
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

    results['task'] = results['dataset'].apply(dataset_to_task)
    clf_results = results[results['task'] == 'classification'].copy()
    reg_results = results[results['task'] == 'regression'].copy()
    clf_interpretability_results = results[results['task'] == 'classifier_interpretability'].copy()

    clf_results['metric'] = clf_results['dataset'].apply(lambda ds: getattr(clf_ds, ds).metric)
    reg_results['metric'] = reg_results['dataset'].apply(lambda ds: getattr(reg_ds, ds).metric)

    return clf_results, reg_results, clf_interpretability_results

def print_results():
    results = get_results_table()
    clf_results, reg_results, _ = split_results_by_task(results)

    print('Classification models global ranking:')
    print(get_model_ranking(clf_results))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification.png', clf_results)

    print('\nClassification models small dataset (<10k examples) ranking')
    big_datasets = ('DefaultCreditCardDataset', 'AdultDataset')
    clf_result_small_ds = clf_results[~clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_small_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification-small-ds.png', clf_result_small_ds)
    
    print('\nClassification models big dataset (>10k examples) ranking')
    big_datasets = ('DefaultCreditCardDataset', 'AdultDataset')
    clf_result_big_ds = clf_results[clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_big_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-classification-big-ds.png', clf_result_big_ds)

    print('\nRegression models global ranking:')
    print(get_model_ranking(reg_results))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression.png', reg_results)

    print('\nRegression models small dataset (<5k examples) ranking')
    big_datasets = ('MerckMolecularActivityDataset', 'BikeSharingDataset', 'SGEMMGPUKernelPerformancesDataset')
    reg_result_small_ds = reg_results[~reg_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(reg_result_small_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression-small-ds.png', reg_result_small_ds)

    print('\nRegression models big dataset (>5k examples) ranking')
    big_datasets = ('MerckMolecularActivityDataset', 'BikeSharingDataset', 'SGEMMGPUKernelPerformancesDataset')
    reg_result_big_ds = reg_results[reg_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(reg_result_big_ds))
    draw_cd_diagram(f'{RESULTS_DIR}/cd-diagram-regression-big-ds.png', reg_result_big_ds)

    print_rankings('Classification', clf_results)
    print_rankings('Regression', reg_results)


def print_rankings(task, results):
    print(f'{task} ranking table')
    results['model_rank'] = results.groupby('dataset')['score'].rank(ascending=False)
    results = results[['dataset', 'model', 'model_rank']]
    results.loc[:, 'dataset'] = results['dataset'].str[:-7]
    results.loc[:, 'model'] = results['model'].str[:-5]
    table = pd.pivot_table(results, values='model_rank', index='model', columns=['dataset'])
    table = table.fillna(table.max())
    table = table.astype(int)

    if task == 'Regression':
        big_ds = ['SGEMMGPUKernelPerformances', 'MerckMolecularActivity', 'BikeSharing']
    else:
        big_ds = ['DefaultCreditCard', 'Adult']
    small_ds = list(set(table.columns) - set(big_ds))

    table['Average'] = table.T.mean()
    table['Average Big Dataset'] = table[big_ds].T.mean()
    table['Average Small Dataset'] = table[small_ds].T.mean()

    if task == 'Regression':
        table.index = ['AB', 'ANN', 'BNN', 'DT', 'GP', 'GB', 'KNN', 'LR', 'RF', 'SVM']
    else:
        table.index = ['AB', 'ANN', 'BNN', 'DT', 'GB', 'KNN', 'LR', 'RF', 'SVM']
    table = table.round(1)
    for col in table.columns:
        table.loc[:, col] = table[col].astype(str)
    table = table.T
    print(table)


def print_all_datasets_results(datasets, results, latex=False):
    dataset_names = results.dataset.unique()
    for dataset_name in dataset_names:
        if latex:
            print("\\begin{figure}\\caption{Results on " + dataset_name[:-7] + " dataset}")
        else:
            print(f'Results on {dataset_name[:-7]} dataset')
        dataset = getattr(datasets, dataset_name)
        table = (results[results.dataset == dataset_name]
        .set_index('model')
        [['score', 'train_time']]
        .sort_values('score', ascending=not dataset.is_metric_maximized)
        .rename(columns={'score': dataset.metric})
        )
        table.index = table.index.rename('')
        table[dataset.metric] = abs(table[dataset.metric])
        if dataset.metric in ('accuracy', 'f1'):
            table[dataset.metric] = (table[dataset.metric] * 100)
        table = table.round(1)

        if latex:
            print(table.to_latex(column_format='|lrr|'))
            print('\\end{figure}')
        else:
            print(table)