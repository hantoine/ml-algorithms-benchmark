import os
from os import makedirs
from os.path import isfile
import urllib.request
import pandas as pd
from scipy.io import arff

class Dataset:
    @classmethod
    def download(cls, workdir):
        #import pdb ; pdb.set_trace()
        makedirs(workdir, exist_ok=True)
        if hasattr(cls, 'url'):
            filepath = os.path.join(workdir, cls.filename)
            cls.download_file(cls.url, filepath)
        elif hasattr(cls, 'urls'):
            for url, filename in zip(cls.urls, cls.filenames):
                filepath = os.path.join(workdir, filename)
                cls.download_file(url, filepath)
        else:
            raise ValueError('No dataset URL specified')


    @staticmethod
    def download_file(url, filepath):
        url = url.replace(' ', '%20')
        urllib.request.urlretrieve(url, filepath)


    @classmethod
    def get_df(cls, workdir, filename):
        dataset_path = os.path.join(workdir, filename)
        if not isfile(dataset_path):
            cls.download(workdir)

        ext = filename.split('.')[-1]
        if ext == 'xls':
            return pd.read_excel(dataset_path, header=[0, 1])
        elif ext == 'dat':
            return pd.read_csv(dataset_path, sep=' ', header=None)
        elif ext == 'data-numeric':
            return pd.read_csv(dataset_path, sep=r'\s+', header=None)
        elif ext == 'NNA':
            return pd.read_csv(dataset_path, sep='\t', header=None)
        elif ext == 'arff':
            data, _ = arff.loadarff(dataset_path)
            return pd.DataFrame(data)
        elif ext == 'data':
            return pd.read_csv(dataset_path, delim_whitespace=True, header=None)
        elif ext == 'txt':
            return pd.read_csv(dataset_path, sep=',', header=None)
        elif ext == 'csv':
            return pd.read_csv(dataset_path, sep=',', header=None)
