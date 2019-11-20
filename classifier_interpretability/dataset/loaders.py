from utils import Dataset
from config import DEFAULT_DATA_DIR

import glob, os
import pickle
import numpy as np
import tarfile
from os.path import isfile


def unpickle(file_path):
    """ Function from documentation
    see: https://www.cs.toronto.edu/~kriz/cifar.html
    return:
        d: dictionary containing the data
    """
    with open(file_path, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_data(filename):
    # each row is a RVB img : [32x32 entries, 32x32, 32x32]
    batch = unpickle(filename)
    return batch[b'data'], batch[b'labels']


class Cifar10Dataset(Dataset):
    filename = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with tarfile.open(dataset_path, 'r:gz') as tf:
            tf.extractall(workdir)

        labels_info = unpickle(os.path.join(workdir, 'cifar-10-batches-py/batches.meta'))
        X_test, y_test = get_data(os.path.join(workdir, 'cifar-10-batches-py/test_batch'))
        train_files = [f for f in glob.glob(os.path.join(workdir, 'cifar-10-batches-py/data_batch*'))]

        X, Y = get_data(train_files[0])
        for f in train_files[1:]:
            x, y = get_data(f)
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))

        return (X, Y), (X_test, y_test), labels_info
