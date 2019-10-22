from os import makedirs
from os.path import isfile
import urllib.request
import urllib.parse
import pandas as pd

class Dataset:
    @classmethod
    def download(cls):
        makedirs('data', exist_ok=True)
        url = cls.url.replace(' ', '%20')
        urllib.request.urlretrieve(url, f'data/{cls.filename}')

class DefaultCreditCardDataset(Dataset):
    filename = 'default of credit card clients.xls'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls'
    @classmethod
    def get(cls):
        if not isfile(f'data/{cls.filename}'):
            cls.download()
        df = pd.read_excel(f'data/{cls.filename}', header=[0, 1])
        
        y = p['Y'].values.ravel()
        X = p[[f'X{i}' for i in range(1, 24)]].values

        # First column is ID and last is label
        cls.features_names = [col[1] for col in p.columns[1:-1]]

        return X, y
