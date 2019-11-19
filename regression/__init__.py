from utils import tune_all_models_on_all_datasets
from regression import datasets, models

def tune_all_models_on_all_regression_datasets(tuning_trials_per_step=5, tuning_time=120):
    tune_all_models_on_all_datasets('regression', datasets.all_datasets, models.all_models,
                                    tuning_trials_per_step, tuning_time)