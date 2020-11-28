import os
import urllib.request
from os import makedirs
from os.path import isfile

import numpy as np
import pandas as pd
from scipy.io import arff


class Dataset:
    @classmethod
    def download(cls, workdir):
        makedirs(workdir, exist_ok=True)
        if hasattr(cls, "url"):
            filepath = os.path.join(workdir, cls.filename)
            cls.download_file(cls.url, filepath)
        elif hasattr(cls, "urls"):
            for url, filename in zip(cls.urls, cls.filenames):
                filepath = os.path.join(workdir, filename)
                cls.download_file(url, filepath)
        else:
            raise ValueError("No dataset URL specified")

    @staticmethod
    def download_file(url, filepath):
        url = url.replace(" ", "%20")
        urllib.request.urlretrieve(url, filepath)

    @classmethod
    def get_min_k_fold_k_value(cls, train_data):
        _, y = train_data
        if type(y) == pd.Series:
            y = y.values
        return np.bincount(y).min()
