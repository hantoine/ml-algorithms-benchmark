from classification import datasets as ds
from shutil import rmtree
import os
from os.path import isdir
from utils import check_dataset

workdir = "test-workdir"
if isdir(workdir):
    rmtree(workdir)


def test_default_credit_card_dataset_loading():
    dataset = ds.DefaultCreditCardDataset.get(workdir)
    check_dataset(dataset)


def test_statlog_australian_dataset_loading():
    dataset = ds.StatlogAustralianDataset.get(workdir)
    check_dataset(dataset)


def test_statlog_german_dataset_loading():
    dataset = ds.StatlogGermanDataset.get(workdir)
    check_dataset(dataset)


def test_adult_dataset_loading():
    dataset = ds.AdultDataset.get(workdir)
    check_dataset(dataset)


def test_retinopathy_dataset_loading():
    dataset = ds.RetinopathyDataset.get(workdir)
    check_dataset(dataset)


def test_thoracic_surgery_dataset_loading():
    dataset = ds.ThoraricSurgeryDataset.get(workdir)
    check_dataset(dataset)


def test_breast_cancer_dataset_loading():
    dataset = ds.BreastCancerDataset.get(workdir)
    check_dataset(dataset)


def test_seismic_bumps_dataset_loading():
    dataset = ds.SeismicBumpsDataset.get(workdir)
    check_dataset(dataset)


def test_steel_plates_faults_dataset_loading():
    dataset = ds.SteelPlatesFaultsDataset.get(workdir)
    check_dataset(dataset)
