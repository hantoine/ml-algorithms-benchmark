import argparse
from classification import tune_all_models_on_all_classification_datasets, \
                            evaluate_all_models_on_all_classification_datasets
from regression import tune_all_models_on_all_regression_datasets, \
                       evaluate_all_models_on_all_regression_datasets

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='commands')
def add_tuning_arguments(parser):
    parser.add_argument('-s', '--tuning_trials_per_step', type=int, default=5)
    parser.add_argument('-t', '--tuning_time', type=int, default=120)
    parser.add_argument('-m', '--max_trials_without_improvement', type=int, default=150)

parser_tune_classification = subparsers.add_parser('classification-tuning')
add_tuning_arguments(parser_tune_classification)
parser_tune_classification.set_defaults(func=tune_all_models_on_all_classification_datasets)

parser_tune_regression = subparsers.add_parser('regression-tuning')
add_tuning_arguments(parser_tune_regression)
parser_tune_regression.set_defaults(func=tune_all_models_on_all_regression_datasets)

parser_train_regression = subparsers.add_parser('regression-evaluation')
parser_train_regression.set_defaults(func=evaluate_all_models_on_all_regression_datasets)

parser_train_classification = subparsers.add_parser('classification-evaluation')
parser_train_classification.set_defaults(func=evaluate_all_models_on_all_classification_datasets)

args = parser.parse_args()

try:
    func = args.func
except AttributeError:
    parser.error("No command specified")

args = vars(args).copy()
del args['func']
func(**args)
