import argparse
from classification import tune_all_models_on_all_classification_datasets, \
    evaluate_all_models_on_all_classification_datasets
from regression import tune_all_models_on_all_regression_datasets, \
    evaluate_all_models_on_all_regression_datasets
from classifier_interpretability import tune_all_models_on_all_classifier_interpretability_datasets, \
    evaluate_all_models_on_all_classifier_interpretability_datasets
from utils import print_results

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='commands')


def add_tuning_arguments(parser):
    parser.add_argument('-s', '--tuning_trials_per_step', type=int, default=5)
    parser.add_argument('-m', '--max_tuning_time', type=int, default=120)
    parser.add_argument('-e', '--max_trials_without_improvement', type=int, default=150)
    parser.add_argument('-t', '--tuning_step_max_time', type=int, default=60)
    parser.add_argument('-d', '--mongo_address', type=str, default=None)


parser_tune_classification = subparsers.add_parser('classification-tuning')
add_tuning_arguments(parser_tune_classification)
parser_tune_classification.set_defaults(func=tune_all_models_on_all_classification_datasets)

parser_tune_regression = subparsers.add_parser('regression-tuning')
add_tuning_arguments(parser_tune_regression)
parser_tune_regression.set_defaults(func=tune_all_models_on_all_regression_datasets)

parser_tune_classifier_interpretability = subparsers.add_parser('classifier-interpretability-tuning')
add_tuning_arguments(parser_tune_classifier_interpretability)
parser_tune_classifier_interpretability.set_defaults(func=tune_all_models_on_all_classifier_interpretability_datasets)

parser_train_classification = subparsers.add_parser('classification-evaluation')
parser_train_classification.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_classification.set_defaults(func=evaluate_all_models_on_all_classification_datasets)

parser_train_regression = subparsers.add_parser('regression-evaluation')
parser_train_regression.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_regression.set_defaults(func=evaluate_all_models_on_all_regression_datasets)

parser_train_classifier_interpretability = subparsers.add_parser('classifier-interpretability-evaluation')
parser_train_classifier_interpretability.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_classifier_interpretability.set_defaults(
    func=evaluate_all_models_on_all_classifier_interpretability_datasets)

parser_summarize_results = subparsers.add_parser('summarize_results')
parser_summarize_results.set_defaults(func=print_results) 


args = parser.parse_args()

try:
    func = args.func
except AttributeError:
    parser.error("No command specified")

args = vars(args).copy()
del args['func']
func(**args)
