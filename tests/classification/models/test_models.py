from classification import models
from hyperopt.pyll.stochastic import sample as sample_hp
from classification.datasets import AdultDataset

def check_prepare_dataset(cls):
    train, _ = AdultDataset.get()
    X, y = train
    cls.prepare_dataset(X, y, AdultDataset.categorical_features)

def test_random_forests_hp_space():
    sample_hp(models.RandomForestsModel.hp_space)

def test_random_forests_prepare():
    check_prepare_dataset(models.RandomForestsModel)