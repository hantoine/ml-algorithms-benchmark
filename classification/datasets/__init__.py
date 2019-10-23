import os
from os.path import isfile
from os import makedirs
import urllib.request
import urllib.parse
import pandas as pd
from scipy.io import arff
import numpy as np

workdir = os.path.join(os.getcwd(), 'data')


class Dataset:
    @classmethod
    def download(cls):
        makedirs(workdir, exist_ok=True)
        if hasattr(cls, 'url'):
            cls.download_file(cls.url, cls.filename)
        elif hasattr(cls, 'urls'):
            for url, filename in zip(cls.urls, cls.filenames):
                cls.download_file(url, filename)
        else:
            raise ValueError('No dataset URL specified')

    @staticmethod
    def download_file(url, filename):
        url = url.replace(' ', '%20')
        urllib.request.urlretrieve(url, os.path.join(workdir, filename))


class DefaultCreditCardDataset(Dataset):
    filename = 'default of credit card clients.xls'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls'

    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download()
        df = pd.read_excel(dataset_path, header=[0, 1])

        y = df['Y'].values.ravel()
        X = df[[f'X{i}' for i in range(1, 24)]].values

        # First column is ID and last is label
        cls.feature_names = [col[1] for col in df.columns[1:-1]]

        return X, y


class SteelPlatesFaultsDataset(Dataset):
    urls = ['https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var']
    filenames = ['Faults.NNA', 'Faults27x7_var']

    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filenames[0])
        if not isfile(dataset_path):
            cls.download()

        df = pd.read_csv(dataset_path, sep='\t', header=None)

        with open(os.path.join(workdir, cls.filenames[1]), 'r') as f:
            cls.feature_names = f.read().strip().split('\n')

        return df


class SeismicBumps(Dataset):
    filename = 'seismic-bumps.arff'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff'

    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download()

        data, meta = arff.loadarff(dataset_path)

        df = pd.DataFrame(data)
        df['class'] = pd.to_numeric(df['class'])

        str_df = df.select_dtypes([np.object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            df[col] = str_df[col]

        return df


# Usage example
if __name__ == '__main__':
    X, y = DefaultCreditCardDataset.get()
    print(f'Features: {DefaultCreditCardDataset.feature_names}')