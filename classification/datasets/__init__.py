import os
from os.path import isfile
from os import makedirs
import urllib.request
import urllib.parse
import pandas as pd
from sklearn.model_selection import train_test_split

workdir = os.path.join(os.getcwd(), 'data')
test_size = 0.25

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

        y = df['Y'][df['Y'].columns[0]] 
        X = df[[f'X{i}' for i in range(1, 24)]]
        X.columns = X.columns.droplevel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)


class StatlogAustralianDataset(Dataset):
    filename = 'australian.dat'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'
    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download()
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)

        
class StatlogGermanDataset(Dataset):
    filename = 'german.data-numeric'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric'
    @classmethod
    def get(cls):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download()
        df = pd.read_csv(dataset_path, sep=r'\s+', header=None)
        
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
    categorical_features = ['workclass', 'education']
    desc_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
    @classmethod
    def get(cls): #TODO see how to deal with seperate tets
        df_train, df_test = cls.get_raw()
        df_train, df_test = cls.parse_dataset(df_train), cls.parse_dataset(df_test) 
        return df_train, df_test
    
    @classmethod
    def get_raw(cls):
        dataset_path = os.path.join(workdir, cls.filenames[0])
        if not isfile(dataset_path):
            cls.download()
        df_train = pd.read_csv(dataset_path, header=None, sep=', ', engine='python')
        df_test = pd.read_csv(os.path.join(workdir, cls.filenames[1]),
                              header=None, skiprows=1, sep=', ', engine='python')
        return df_train, df_test
    
    @classmethod
    def parse_dataset(cls, df):
        X, y = df[df.columns[:14]], df[df.columns[14]]
        X.columns = cls.feature_names
        return X, y

# Usage example
if __name__ == '__main__':
    X, y = DefaultCreditCardDataset.get()
    print(f'Features: {DefaultCreditCardDataset.feature_names}')

