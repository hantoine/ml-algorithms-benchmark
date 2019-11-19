from copy import deepcopy
import warnings
from datetime import timedelta
from sklearn.exceptions import ConvergenceWarning

from classification import datasets as ds
from classification import models
from utils import tune_hyperparams


HP_TUNING_STEP_TRIALS = 5
HP_TUNE_MAX_TIME = 120

warnings.filterwarnings("ignore", category=ConvergenceWarning)

minimum_runtime = HP_TUNE_MAX_TIME * len(models.all_models) * len(ds.all_datasets)
print(f'Expected minimum runtime: {timedelta(seconds=minimum_runtime)}')

for dataset in ds.all_datasets:
    print(f'Dataset: {dataset.__name__}')
    train, test = dataset.get()
    for model in models.all_models:
        print(f'Model: {model.__name__}')
        try:
            # Pre-processings are only applied on the data used by this model
            train_prepared, test_prepared = deepcopy((train, test))
            train_prepared, test_prepared = \
                model.prepare_dataset(train_prepared, test_prepared, dataset.categorical_features)
            best_hp = tune_hyperparams('classification', dataset, model, train_prepared,
                                        HP_TUNING_STEP_TRIALS, HP_TUNE_MAX_TIME)
        except MemoryError:
            print('Memory requirements for this model with this dataset too high')