import argparse
from classification import tune_all_models_on_all_classification_datasets
from regression import tune_all_models_on_all_regression_datasets

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title='commands')

parser_tune_classification = subparsers.add_parser('classification-tuning')
parser_tune_classification.add_argument('-s', '--tuning_trials_per_step', type=int, default=5)
parser_tune_classification.add_argument('-t', '--tuning_time', type=int, default=120)
parser_tune_classification.set_defaults(func=tune_all_models_on_all_classification_datasets)

parser_tune_regression = subparsers.add_parser('regression-tuning')
parser_tune_regression.add_argument('-s', '--tuning_trials_per_step', type=int, default=5)
parser_tune_regression.add_argument('-t', '--tuning_time', type=int, default=120)
parser_tune_regression.set_defaults(func=tune_all_models_on_all_regression_datasets)

args = parser.parse_args()

try:
    func = args.func
except AttributeError:
    parser.error("No command specified")

args = vars(args).copy()
del args['func']
func(**args)
