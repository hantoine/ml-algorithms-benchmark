from glob import glob
from config import RESULTS_DIR
import json
import re
import pandas as pd
from classification import datasets as clf_ds
from regression import datasets as reg_ds

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

def print_results(latex=False):
    results = get_results_table()
    # pd.set_option('display.max_rows', -1)
    # pd.set_option('display.max_colwidth', -1)
    clf_results, reg_results, _ = split_results_by_task(results)

    print('Classification models global ranking:')
    print(get_model_ranking(clf_results))

    print('Classification models small dataset (<10k examples) ranking')
    big_datasets = ('DefaultCreditCardDataset', 'AdultDataset')
    clf_result_small_ds = clf_results[~clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_small_ds))
    
    print('Classification models big dataset (>10k examples) ranking')
    big_datasets = ('DefaultCreditCardDataset', 'AdultDataset')
    clf_result_big_ds = clf_results[clf_results.dataset.isin(big_datasets)].copy()
    print(get_model_ranking(clf_result_big_ds))

    # Regression to be done

    print(results)