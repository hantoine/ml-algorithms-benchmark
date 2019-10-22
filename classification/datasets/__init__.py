from os import makedirs
from os.path import isfile
import urllib.request
import urllib.parse
import pandas as pd

workdir = os.path.join(os.getcwd(), 'data')

class Dataset:
    @classmethod
    def download(cls):
        makedirs(workdir, exist_ok=True)
        url = cls.url.replace(' ', '%20')
        urllib.request.urlretrieve(url, os.path.join(workdir, cls.filename))

class DefaultCreditCardDataset(Dataset):
    filename = 'default of credit card clients.xls'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls'
    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download()
        df = pd.read_excel(dataset_path, header=[0, 1])
        
        y = p['Y'].values.ravel()
        X = p[[f'X{i}' for i in range(1, 24)]].values

        # First column is ID and last is label
        cls.features_names = [col[1] for col in p.columns[1:-1]]

        return X, y
