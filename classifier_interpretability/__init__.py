from utils import tune_all_models_on_all_datasets, train_all_models_on_all_datasets
from classifier_interpretability import datasets, models


def tune_all_models_on_all_classifier_interpretability_datasets(tuning_trials_per_step=5, tuning_time=120,
                                                                max_trials_without_improvement=150,
                                                                tuning_step_max_time=60):
    tune_all_models_on_all_datasets('classification', datasets.all_datasets, models.all_models,
                                    tuning_trials_per_step, tuning_time,
                                    max_trials_without_improvement, tuning_step_max_time)


def evaluate_all_models_on_all_classifier_interpretability_datasets(max_training_time=180):
    train_all_models_on_all_datasets(datasets.all_datasets, models.all_models, max_training_time)
