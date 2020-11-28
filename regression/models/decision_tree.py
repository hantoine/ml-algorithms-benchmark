from sklearn.tree import DecisionTreeRegressor
from utils import TreeBasedModel
from config import RANDOM_STATE
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class DecisionTreeModel(TreeBasedModel):
    @staticmethod
    def build_estimator(args, train_data=None):
        return DecisionTreeRegressor(random_state=RANDOM_STATE, presort=True, **args)

    hp_space = {
        "criterion": hp.choice("criterion", ["mse", "friedman_mse", "mae"]),
        "max_depth": hp.pchoice(
            "max_depth_enabled",
            [
                (0.7, None),
                (0.3, 1 + scope.int(hp.qlognormal("max_depth", np.log(30), 0.5, 3))),
            ],
        ),
        "splitter": hp.choice("splitter_str", ["best", "random"]),
        "max_features": hp.pchoice(
            "max_features_str",
            [
                (0.2, "sqrt"),  # most common choice.
                (0.1, "log2"),  # less common choice.
                (0.1, None),  # all features, less common choice.
                (0.6, hp.uniform("max_features_str_frac", 0.0, 1.0)),
            ],
        ),
        "min_samples_split": scope.int(hp.quniform("min_samples_split_str", 2, 10, 1)),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf_enabled",
            [
                1,
                scope.int(
                    hp.qloguniform("min_samples_leaf", np.log(1.5), np.log(50.5), 1)
                ),
            ],
        ),
    }
