import os
from os.path import isfile
import urllib.parse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import patoolib  # for rar files
from zipfile import ZipFile
import time
from scipy.io import arff

from utils import Dataset, test_size, random_state
from config import DEFAULT_DATA_DIR


class WhiteWineQualityDataset(Dataset):
    filename = 'winequality-red.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=';')
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class RedWineQualityDataset(Dataset):
    filename = 'winequality-white.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=';')
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class CommunitiesAndCrimeDataset(Dataset):
    filename = 'communities.data'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=',', header=None)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class QsarAquaticToxicityDataset(Dataset):
    filename = 'qsar_aquatic_toxicity.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00505//qsar_aquatic_toxicity.csv'
    features = ['TPSA', 'SAacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040', 'LC50']

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        df.columns = cls.features
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)


class ParkinsonMultipleSoundRecordingDataset(Dataset):
    filename = 'Parkinson_Multiple_Sound_Recording.rar'
    filenames = ['train_data.txt',
                 'test_data.txt']

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00301/Parkinson_Multiple_Sound_Recording.rar'

    feature_names = ['subject_id',
                     'jitter_local', 'jitter_local_abs', 'jitter_rap',
                     'jitter_ppq5', 'jitter_ddp', 'shimmer_local',
                     'shimmer_local_db', 'shimmer_apq3', 'shimmer_apq5',
                     'shimmer_apq11', 'shimmer_dda', 'AC', 'NTH', 'HTN',
                     'median_pitch', 'mean_pitch', 'std', 'minimum_pitch',
                     'maximum_pitch', 'nb_pulses', 'nb_periods', 'mean_period',
                     'std', 'frac_local_unvoiced_frames', 'nb_voice_breaks',
                     'degree_voice_breaks']

    @classmethod
    def parse_dataset(cls, df):
        X, y = df[df.columns[:27]], df[df.columns[26]]
        X.columns = cls.feature_names
        return X, y

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        patoolib.extract_archive(dataset_path, outdir=workdir)
        time.sleep(2)

        train_df = cls.get_df(workdir, cls.filenames[0])
        test_df = cls.get_df(workdir, cls.filenames[1])
        train_df = train_df.drop([27])

        X_train, y_train = cls.parse_dataset(train_df)
        X_test, y_test = cls.parse_dataset(test_df)
        return (X_train, y_train), (X_test, y_test)


class BikeSharingDataset(Dataset):
    filename = 'Bike-Sharing-Dataset.zip'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with ZipFile(dataset_path, 'r') as zipfile:
            zipfile.extractall(workdir)
        df = pd.read_csv(os.path.join(workdir, 'hour.csv'), sep=',')
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class ConcreteCompressiveStrengthDataset(Dataset):
    filename = 'Concrete_Data.xls'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_excel(dataset_path)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)
