from sklearn.linear_model import LogisticRegression
from utils import NonTreeBasedModel
from config import RANDOM_STATE
from hyperopt import hp
import numpy as np


class LRModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, train_data=None, test=False):
        return LogisticRegression(
            random_state=RANDOM_STATE,
            solver="saga",
            max_iter=500,  # increased because Convergence was not always reached
            **args
        )

    hp_space = {
        "penalty": "elasticnet",
        "C": hp.loguniform("C", np.log(1e-3), np.log(1e3)),
        "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
        "class_weight": hp.pchoice("class_weight", [(0.5, None), (0.5, "balanced"),]),
        "multi_class": "multinomial",
    }
