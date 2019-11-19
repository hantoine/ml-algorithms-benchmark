from sklearn.linear_model import LogisticRegression
from utils import NonTreeBasedModel, random_state
from hyperopt import hp
import numpy as np


class LRModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, test=False):
        return LogisticRegression(
            random_state=random_state,
            solver='saga',
            **args
        )

    hp_space = {
        'penalty': 'elasticnet',
        'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
        'class_weight': hp.pchoice('class_weight', [
            (0.5, None),
            (0.5, 'balanced'),
        ]),
        'multi_class': 'multinomial'
    }
