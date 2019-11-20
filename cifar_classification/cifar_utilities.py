import glob, os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    print(f'file {filename} processed')
    batch = unpickle(filename)
    return batch[b'data'], batch[b'labels']


def get_data_from_files(data_path):
    os.chdir(data_path)
    labels_info = dict()
    files = [f for f in glob.glob("*")]
    train_files = [f for f in glob.glob("data_batch*")]

    for f in files:
        if f == "batches.meta":
            print(f'file {f} processed')
            labels_info = unpickle(f)
        elif f == "test_batch":
            X_test, y_test = get_data(f)

    X, Y = get_data(train_files[0])
    for f in train_files[1:]:
        print(f'file {f} processed')
        x, y = get_data(f)
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))

    return (X,Y), (X_test, y_test), labels_info
