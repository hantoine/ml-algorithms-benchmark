import regression.datasets.loaders as ld
from shutil import rmtree
import os
from os.path import isdir
import pandas as pd
from utils import check_dataset


workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)


def test_parkinson_multiple_sound_recording():
    dataset = ld.ParkinsonMultipleSoundRecording.get(workdir)
    check_dataset(dataset)


"""def test_merck_molecular_activity_challenge():
    dataset = ld.MerckMolecularActivityChallenge.get(workdir)
    check_dataset(dataset)"""


def test_sqar_aquatic_toxicity():
    dataset = ld.QsarAquaticToxicity.get(workdir)
    check_dataset(dataset)
