import regression.datasets.loaders as ld
from shutil import rmtree
import os
from os.path import isdir
import pandas as pd
from utils import check_dataset


workdir = os.path.join('tests/regression/datasets/test-workdir')
if isdir(workdir):
    rmtree(workdir)


def test_parkinson_multiple_sound_recording():
    dataset = ld.ParkinsonMultipleSoundRecording.get(workdir)
    check_dataset(dataset)
