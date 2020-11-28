from numpy.random import RandomState
from regression import models
from hyperopt.pyll.stochastic import sample as sample_hp
from regression import datasets as ds
import os
from os.path import isdir
from shutil import rmtree

workdir = "test-workdir"


def test_random_forests_hp_space():
    sample_hp(models.RandomForestsModel.hp_space)


def test_random_forest_training():
    model = models.RandomForestsModel
    hyperparams = sample_hp(models.RandomForestsModel.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_svm_training():
    model = models.SVMModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams, test=True)
        X, y, *_ = train
        estimator.fit(X, y)


def test_linear_regression_training():
    model = models.LinearRegressionModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams, test=True)
        X, y, *_ = train
        estimator.fit(X, y)


def test_neural_network_training():
    model = models.NeuralNetworkModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_gaussian_process_training():
    model = models.GaussianProcessModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        try:
            print(dataset.__name__)
            train, test = dataset.get()
            train, test = model.prepare_dataset(
                train, test, dataset.categorical_features
            )
            estimator = model.build_estimator(hyperparams)
            X, y, *_ = train
            estimator.fit(X, y)
        except MemoryError:
            # This model has high memory requirements and cannot be used on some big datasets
            continue


def test_decision_tree_training():
    model = models.DecisionTreeModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_ada_boost_training():
    model = models.AdaBoostModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_knn_training():
    model = models.KNearestNeighborsModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_gradient_boosting_training():
    model = models.GradientBoostingModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)
