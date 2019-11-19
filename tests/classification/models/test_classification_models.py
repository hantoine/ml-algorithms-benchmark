from numpy.random import RandomState
from classification import models
from hyperopt.pyll.stochastic import sample as sample_hp
from classification import datasets as ds

workdir = 'test-workdir'


def check_prepare_dataset(cls):
    train, test = ds.AdultDataset.get(workdir)
    cls.prepare_dataset(train, test, ds.AdultDataset.categorical_features)


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


def test_random_forest_training():
    model = models.RandomForestsModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)


def test_svm_training():
    model = models.SVMModel
    hyperparams = sample_hp(models.SVMModel.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams, test=True)
        X, y, *_ = train
        estimator.fit(X, y)


def test_lr_training():
    model = models.LRModel
    hyperparams = sample_hp(model.hp_space, rng=RandomState(1))
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
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


def test_gaussian_nb_training():
    model = models.GaussianNBModel
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(None)
        X, y, *_ = train
        estimator.fit(X, y)
