import classification.datasets.loaders as ld
from shutil import rmtree
import os
from os.path import isdir
from utils import check_dataset


workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)


def test_default_credit_card_dataset_loading():
    dataset = ld.DefaultCreditCardDataset.get(workdir)
    check_dataset(dataset)


def test_statlog_australian_dataset_loading():
    dataset = ld.StatlogAustralianDataset.get(workdir)
    check_dataset(dataset)


def test_statlog_german_dataset_loading():
    dataset = ld.StatlogGermanDataset.get(workdir)
    check_dataset(dataset)


"""def test_adult_dataset_loading():
    dataset = ld.AdultDataset.get()
    check_dataset(dataset)"""


def test_retinopathy_dataset_loading():
    dataset = ld.Retinopathy.get(workdir)
    check_dataset(dataset)


def test_thoracic_surgery_dataset_loading():
    dataset = ld.ThoraricSurgery.get(workdir)
    check_dataset(dataset)


def test_breast_cancer_dataset_loading():
    dataset = ld.BreastCancer.get(workdir)
    check_dataset(dataset)
