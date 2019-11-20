from utils import Dataset
from config import DEFAULT_DATA_DIR

import glob, os
import pickle
import numpy as np
import tarfile
from os.path import isfile


class Cifar10Dataset(Dataset):
    filename = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    categorical_features = []
    metric = 'accuracy'

    @staticmethod
    def unpickle(file_path):
        """ Function from documentation
        see: https://www.cs.toronto.edu/~kriz/cifar.html
        return:
            d: dictionary containing the data
        """
        with open(file_path, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with tarfile.open(dataset_path, 'r:gz') as tf:
            tf.extractall(workdir)

        cls.labels_info = cls.unpickle(os.path.join(workdir, 'cifar-10-batches-py/batches.meta'))
        test_data = cls.unpickle(os.path.join(workdir, 'cifar-10-batches-py/test_batch'))
        X_test, y_test = np.array(test_data[b'data']), np.array(test_data[b'labels'])
        train_files = [f for f in glob.glob(os.path.join(workdir, 'cifar-10-batches-py/data_batch*'))]

        first_batch = cls.unpickle(train_files[0])
        X, Y = first_batch[b'data'], first_batch[b'labels']
        for f in train_files[1:]:
            batch = cls.unpickle(f)
            x, y = batch[b'data'], batch[b'labels']
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))

        print(type(Y), type(y_test))

        return (X, Y), (X_test, y_test)
