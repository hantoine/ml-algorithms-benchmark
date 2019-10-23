import os
from os.path import isfile
from os import makedirs
import urllib.request
import urllib.parse
import pandas as pd

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

# Usage example
if __name__ == '__main__':
    X, y = DefaultCreditCardDataset.get()
    print(f'Features: {DefaultCreditCardDataset.feature_names}')

