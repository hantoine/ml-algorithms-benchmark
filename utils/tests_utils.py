import pandas as pd
import numpy as np


def check_dataset(dataset):
    assert len(dataset) == 2
    train, test = dataset
    assert len(train) >= 2 and len(test) == 2
    x_train, y_train, *others = train
    x_test, y_test = test
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)
    assert len(x_test) < len(x_train)
    assert type(y_train) == type(y_test)
    assert (type(y_train) == pd.core.series.Series
            or type(y_train) == np.ndarray
            or type(y_train) == pd.DataFrame)
    assert (type(x_train) == type(x_test) == pd.core.frame.DataFrame
            or type(x_train) == type(x_test) == np.ndarray)
    assert x_train.shape[1] == x_test.shape[1]
    assert hasattr(dataset, 'categorical_features')
    assert type(dataset.categorical_features) == list
    assert hasattr(dataset, 'metric')
    assert type(dataset.metric) == str

    if len(others) != 0:
        assert len(others) == 1
        groups_train = others[0]
        assert type(groups_train) == pd.core.series.Series or type(groups_train) == np.ndarray
        assert len(groups_train) == len(y_train)
