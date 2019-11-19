from sklearn.svm import SVR
from utils import NonTreeBasedModel
from hyperopt import hp
import numpy as np


class SVMModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, test=False):
        return SVR(
            # Prevent very long training time for some hyper-parameters
            max_iter=int(1e3) if test else int(2e6),
            # Use 4GB of cache to speed up
            cache_size=4096,
            **args
        )

    # Hyper-parameters distributions
    C_hp = hp.loguniform('C', np.log(1e-5), np.log(1e4))
    gamma_hp = hp.pchoice('gamma_choice', [
        (0.3, 'auto'),
        (0.3, 'scale'),
        (0.2, hp.loguniform('gamma', np.log(1e-4), np.log(1e2)))
    ])
    class_weight_hp = hp.pchoice('class_weight', [
        (0.5, None),
        (0.5, 'balanced'),
    ])
    coef0_hp = hp.pchoice('coef0_null', [
        (0.3, 0.0),
        (0.7, hp.loguniform('coef0', np.log(1e-3), np.log(1e3)))
    ])
    epsilon_hp = hp.loguniform('epsilon', np.log(1e-2), np.log(1e2))

    # Defintions of hyper-parameters spaces for each kernel
    linear_hp_space = {
        'kernel': 'linear',
        'C': C_hp,
        'epsilon': epsilon_hp
    }
    poly_hp_space = {
        'kernel': 'poly',
        'C': C_hp,
        'degree': hp.quniform('degree', 1.5, 6.5, 1),
        'gamma': gamma_hp,
        'coef0': coef0_hp,
        'epsilon': epsilon_hp
    }
    rbf_hp_space = {
        'kernel': 'rbf',
        'C': C_hp,
        'gamma': gamma_hp,
        'epsilon': epsilon_hp
    }
    sigmoid_hp_space = {
        'kernel': 'sigmoid',
        'C': C_hp,
        'gamma': gamma_hp,
        'coef0': coef0_hp,
        'epsilon': epsilon_hp
    }

    # Final hyper-parameters space
    hp_space = hp.pchoice('kernel', [
        (0.5, rbf_hp_space),
        (0.2, linear_hp_space),
        (0.2, poly_hp_space),
        (0.1, sigmoid_hp_space)
    ])
