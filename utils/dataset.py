import os
from os import makedirs
import urllib.request

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
