import numpy as np
import pandas as pd

def get_min_k_fold_k_value(train_data):
    _, y = train_data
    if type(y) == pd.Series:
        y = y.values
    return np.bincount(y).min()