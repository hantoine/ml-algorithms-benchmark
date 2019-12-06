import argparse
from classification import tune_all_models_on_all_classification_datasets, \
    evaluate_all_models_on_all_classification_datasets
from regression import tune_all_models_on_all_regression_datasets, \
    evaluate_all_models_on_all_regression_datasets
from classifier_interpretability import tune_all_models_on_all_clf_interpret_datasets, \
                                        evaluate_all_models_on_all_clf_interpret_datasets, \
                                        generate_interpretation_viz
from utils import print_results

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(metavar='command')


def add_tuning_arguments(parser):
    parser.add_argument('-s', '--tuning_trials_per_step', type=int, default=5)
    parser.add_argument('-m', '--max_tuning_time', type=int, default=120)
    parser.add_argument('-e', '--max_trials_without_improvement', type=int, default=150)
    parser.add_argument('-t', '--tuning_step_max_time', type=int, default=60)
    parser.add_argument('-d', '--mongo_address', type=str, default=None)


parser_tune_classification = subparsers.add_parser('classification-tuning', help='Tuning of all the algorithms on all the classification datasets')
add_tuning_arguments(parser_tune_classification)
parser_tune_classification.set_defaults(func=tune_all_models_on_all_classification_datasets)

parser_tune_regression = subparsers.add_parser('regression-tuning', help='Tuning of all the algorithms on all the regression datasets')
add_tuning_arguments(parser_tune_regression)
parser_tune_regression.set_defaults(func=tune_all_models_on_all_regression_datasets)

parser_tune_clf_interpret = subparsers.add_parser('classifier-interpretability-tuning', help='Tuning of the decision tree and the custom CNN on the CIFIAR-10 dataset')
add_tuning_arguments(parser_tune_clf_interpret)
parser_tune_clf_interpret.set_defaults(func=tune_all_models_on_all_clf_interpret_datasets)

parser_train_classification = subparsers.add_parser('classification-evaluation', help='Evaluation of the performances of all the algorithms on the test set of all the classification datasets')
parser_train_classification.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_classification.set_defaults(func=evaluate_all_models_on_all_classification_datasets)

parser_train_regression = subparsers.add_parser('regression-evaluation', help='Evaluation of the performances of all the algorithms on the test set of all the regression datasets')
parser_train_regression.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_regression.set_defaults(func=evaluate_all_models_on_all_regression_datasets)

parser_train_clf_interpret = subparsers.add_parser('classifier-interpretability-evaluation', help='Evaluation of the performances of all the decision tree and the custom CNN on the test set of CIFAR-10 dataset')
parser_train_clf_interpret.add_argument('-m', '--max_training_time', type=int, default=180)
parser_train_clf_interpret.set_defaults(func=evaluate_all_models_on_all_clf_interpret_datasets)

parser_clf_interpret_generate_vizualizations = \
    subparsers.add_parser('classifier-interpretability-interpretation', help='Generation of the interpretations vizualization (decision tree graph, activation maximization and class activation mapping results)')
parser_clf_interpret_generate_vizualizations.add_argument('-i', '--image_index', type=int, default=11)
parser_clf_interpret_generate_vizualizations.set_defaults(func=generate_interpretation_viz)

parser_summarize_results = subparsers.add_parser('summarize_results', help='Generate results tables and visualizations')
parser_summarize_results.set_defaults(func=print_results) 


args = parser.parse_args()

try:
    func = args.func
except AttributeError:
    parser.print_help()
    exit(1)

args = vars(args).copy()
del args['func']
func(**args)
