from classification import models
from hyperopt.pyll.stochastic import sample as sample_hp
from classification.datasets import AdultDataset
import os
from os.path import isdir
from shutil import rmtree

workdir = os.path.join('tests/classification/models/test-workdir')
if isdir(workdir):
    rmtree(workdir)


def check_prepare_dataset(cls):
    train, _ = AdultDataset.get(workdir)
    X, y = train
    cls.prepare_dataset(X, y, AdultDataset.categorical_features)


def test_random_forests_hp_space():
    sample_hp(models.RandomForestsModel.hp_space)


def test_random_forests_prepare():
    check_prepare_dataset(models.RandomForestsModel)
