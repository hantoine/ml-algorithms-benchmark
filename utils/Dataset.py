import os
from os import makedirs
import urllib.request

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
