import pandas as pd
import numpy as np
from utils import encode_feature_as_cyclical

def test_encode_feature_as_cyclical():
    X = pd.DataFrame({'feature': [1, 2, 3, 4]})
    X_expected = pd.DataFrame({'feature_cos': [0, -1, 0, 1],
                               'feature_sin': [1, 0, -1, 0]})
    X = encode_feature_as_cyclical(X, 'feature', 4)
    assert len(X.columns) == 2
    assert 'feature_cos' in X.columns.values
    assert 'feature_sin' in X.columns.values
    assert np.allclose(X['feature_cos'].values, X_expected['feature_cos'].values)
    assert np.allclose(X['feature_sin'].values, X_expected['feature_sin'].values)