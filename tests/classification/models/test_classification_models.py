from classification import models
from hyperopt.pyll.stochastic import sample as sample_hp
from classification import datasets as ds
import os
from os.path import isdir
from shutil import rmtree

workdir = 'test-workdir'

def check_prepare_dataset(cls):
    train, test = ds.AdultDataset.get(workdir)
    cls.prepare_dataset(train, test, ds.AdultDataset.categorical_features)

def test_random_forests_hp_space():
    sample_hp(models.RandomForestsModel.hp_space)


def test_random_forests_prepare():
    check_prepare_dataset(models.RandomForestsModel)

def test_random_forest_training():
    for dataset in ds.all_datasets:
        train, test = dataset.get()
        model = models.RandomForestsModel
        hyperparams = sample_hp(models.RandomForestsModel.hp_space)
        train, test = model.prepare_dataset(train, test, dataset.categorical_features)
        estimator = model.build_estimator(hyperparams)
        X, y, *_ = train
        estimator.fit(X, y)