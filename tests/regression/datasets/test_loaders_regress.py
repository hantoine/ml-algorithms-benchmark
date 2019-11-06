from regression import datasets as ds
from shutil import rmtree
import os
from os.path import isdir
import pandas as pd
from utils import check_dataset

workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)


def test_white_wine_quality():
    dataset = ds.WhiteWineQualityDataset.get(workdir)
    check_dataset(dataset)

def test_red_whine_quality():
    dataset = ds.RedWineQualityDataset.get(workdir)
    check_dataset(dataset)


def test_communities_and_crime():
    dataset = ds.CommunitiesAndCrimeDataset.get(workdir)
    check_dataset(dataset)


def test_sqar_aquatic_toxicity():
    dataset = ds.QsarAquaticToxicityDataset.get(workdir)
    check_dataset(dataset)


def test_parkinson_multiple_sound_recording():
    dataset = ds.ParkinsonMultipleSoundRecordingDataset.get(workdir)
    check_dataset(dataset)


def test_facebook_metrics():
    dataset = ds.FacebookMetricsDataset.get(workdir)
    check_dataset(dataset)


def test_bike_sharing():
    dataset = ds.BikeSharingDataset.get(workdir)
    check_dataset(dataset)


def test_student_performance():
    dataset = ds.StudentPerformanceDataset.get(workdir)
    check_dataset(dataset)


def test_concrete_compressive_stength():
    dataset = ds.ConcreteCompressiveStrengthDataset.get(workdir)
    check_dataset(dataset)


def test_sgemm_gpu_kernel_performance():
    dataset = ds.SGEMMGPUKernelPerformancesDataset.get(workdir)
    check_dataset(dataset)
