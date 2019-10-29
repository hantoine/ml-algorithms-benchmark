from regression import datasets as ds
from shutil import rmtree
import os
from os.path import isdir
import pandas as pd
from utils import check_dataset


workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)


def test_parkinson_multiple_sound_recording():
    dataset = ds.ParkinsonMultipleSoundRecordingDataset.get(workdir)
    check_dataset(dataset)


"""def test_merck_molecular_activity_challenge():
    dataset = ld.MerckMolecularActivityChallengeDataset.get(workdir)
    check_dataset(dataset)"""


def test_sqar_aquatic_toxicity():
    dataset = ds.QsarAquaticToxicityDataset.get(workdir)
    check_dataset(dataset)
