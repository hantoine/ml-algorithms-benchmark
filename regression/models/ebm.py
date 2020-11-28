from interpret.glassbox import ExplainableBoostingRegressor
from utils import TreeBasedModel
from config import RANDOM_STATE
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np


class ExplainableBoostingMachineModel(TreeBasedModel):
    @staticmethod
    def build_estimator(args, train_data=None):
        feature_names = [f"featur_{i}" for i in range(train_data[0].shape[1])]
        return ExplainableBoostingRegressor(
            random_state=RANDOM_STATE, feature_names=feature_names, **args
        )

    hp_space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(1.0)),
        "max_bins": scope.int(hp.quniform("max_bins", 20, 400, 3)),
        "max_leaves": scope.int(hp.loguniform("max_leaves", np.log(2), np.log(100))),
        "interactions": scope.int(hp.uniform("interactions", 0, 3)),
    }
