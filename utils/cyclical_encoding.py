from math import pi
import numpy as np


def encode_feature_as_cyclical(X, feature_name, maximum_value):
    scaled_feature = 2 * pi * X[feature_name] / maximum_value
    # Use assign to prevent SettingWithCopyWarning
    new_cols = {
        f"{feature_name}_cos": np.cos(scaled_feature),
        f"{feature_name}_sin": np.sin(scaled_feature),
    }
    X = X.assign(**new_cols)
    return X.drop(feature_name, axis="columns")
