import os
from os.path import isfile
import urllib.parse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

from utils import Dataset, test_size, random_state
from config import DEFAULT_DATA_DIR
# import pdb ; pdb.set_trace()


class DefaultCreditCardDataset(Dataset):
    filename = 'default of credit card clients.xls'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        y = df['Y'][df['Y'].columns[0]]
        X = df[[f'X{i}' for i in range(1, 24)]]
        X.columns = X.columns.droplevel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)


class StatlogAustralianDataset(Dataset):
    filename = 'australian.dat'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)


class StatlogGermanDataset(Dataset):
    filename = 'german.data-numeric'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)


class AdultDataset(Dataset):
    filenames = ['adult.data', 'adult.test']
    urls = ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test']
    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    desc_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df_train, df_test = cls.get_raw(workdir)
        X_train, y_train = cls.parse_dataset(df_train)
        X_test, y_test = cls.parse_dataset(df_test)

        le = LabelEncoder().fit(y_train)
        y_train = le.transform(y_train)
        y_test = y_test.str[:-1]  # Additional . at the end of labels in test
        y_test = le.transform(y_test)

        return (X_train, y_train), (X_test, y_test)

    @classmethod
    def get_raw(cls, workdir):
        dataset_path = os.path.join(workdir, cls.filenames[0])
        if not isfile(dataset_path):
            cls.download(workdir)
        df_train = pd.read_csv(os.path.join(workdir, cls.filenames[0]),
                               header=None, skiprows=1, sep=', ', engine='python')
        df_test = pd.read_csv(os.path.join(workdir, cls.filenames[1]),
                              header=None, skiprows=1, sep=', ', engine='python')
        return df_train, df_test

    @classmethod
    def parse_dataset(cls, df):
        X, y = df[df.columns[:14]], df[df.columns[14]]
        X.columns = cls.feature_names
        return X, y


class SteelPlatesFaultsDataset(Dataset):
    urls = ['https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var']
    filenames = ['Faults.NNA', 'Faults27x7_var']

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filenames[0])

        dataset_path = os.path.join(workdir, cls.filenames[1])
        if not isfile(dataset_path):
            cls.download(workdir)
        with open(dataset_path, 'r') as f:
            cls.feature_names = f.read().strip().split('\n')

        X = df[df.columns[:27]]
        y = df[df.columns[27:]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class SeismicBumpsDataset(Dataset):
    filename = 'seismic-bumps.arff'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        df['class'] = pd.to_numeric(df['class'])

        str_df = df.select_dtypes([np.object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            df[col] = str_df[col]

        X = df.drop(columns=['class'])
        y = df.loc[:, 'class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class YeastDataset(Dataset):
    filename = 'yeast.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class RetinopathyDataset(Dataset):
    filename = 'messidor_features.arff'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'

    feature_names = ['quality',
            'pre-screening_label'] + list(range(2, 16)) + [
            'dist_betw_centers',
            'od_diameter',
            'AM_FM_label',
            'class']

    @classmethod
    def preprocess(cls, df):
        df.columns = cls.feature_names
        df['class'] = pd.to_numeric(df['class'])
        df.drop(columns=['quality'])
        return df.drop(columns=['class']), df.loc[:,'class']

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        X, y = cls.preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class ThoraricSurgeryDataset(Dataset):
    filename = 'ThoraricSurgery.arff'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff'

    @classmethod
    def preprocess(cls, df):
        return df.iloc[:,:-1], df.iloc[:,-1]

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        X, y = cls.preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)


class BreastCancerDataset(Dataset):
    filename = 'breast-cancer-wisconsin.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df = cls.get_df(workdir, cls.filename)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)
