from operator import itemgetter
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

class CategoryEncoder:
    def __init__(self, categorical_features, method='onehot'):
        self.categorical_features = categorical_features
        if method not in ('onehot', 'ordinal', 'sorted_ordinal'):
            raise ValueError(f'Invalid method: {method}')
        self.method = method

    @staticmethod
    def get_optimal_category_ordering(X, y, feature_name):
        p_pos_per_category = [(category, y[X[feature_name] == category].mean())
                        for category in X[feature_name].unique()]
        return list(map(itemgetter(0), sorted(p_pos_per_category, key=itemgetter(1))))

    def fit(self, X, y):
        if self.method == 'onehot':
            encoder = OneHotEncoder(dtype=np.int, categories='auto')
        elif self.method == 'sorted_ordinal':
            # Converting to str (OrdinalEncoder does not supported non-sorted numerical categories)
            X[self.categorical_features] = X[self.categorical_features].astype(str)
            categories = [self.get_optimal_category_ordering(X, y, name) for name in self.categorical_features]
            encoder = OrdinalEncoder(categories=categories, dtype=np.int)
        else: # method == 'ordinal'
            encoder = OrdinalEncoder(dtype=np.int)
        self.encoder = ColumnTransformer([('encoder', encoder, self.categorical_features)],
                                         remainder='passthrough')
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'encoder'):
            raise ValueError('The tree ordinal encoder has not been fitted first')
        if self.method == 'sorted_ordinal':
            X[self.categorical_features] = X[self.categorical_features].astype(str)
        X_enc = self.encoder.transform(X)
        if y is not None:
            return X_enc, y
        else:
            return X_enc

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
