from classifier_interpretability import datasets as ds
from shutil import rmtree
from os.path import isdir
from utils import check_dataset

workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)


def test_default_credit_card_dataset_loading():
    dataset = ds.Cifar10Dataset.get(workdir)
    check_dataset(dataset)
