import numpy as np
import torch
import torch.nn.functional as F
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.datasets import make_classification
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
from skorch.dataset import CVSplit
from torch import nn
from torch.optim import lr_scheduler

from config import RANDOM_STATE
from utils import NonTreeBasedModel


class AdvancedNeuralNetworkModel(NonTreeBasedModel):
    @classmethod
    def prepare_dataset(cls, train_data, test_data, categorical_features):
        (X_train, y_train, *other), (X_test, y_test) = super(
            AdvancedNeuralNetworkModel, cls
        ).prepare_dataset(train_data, test_data, categorical_features)
        return (
            (
                X_train.astype(np.float32),
                y_train.astype(np.float32).reshape((-1, 1)),
                *other,
            ),
            (X_test.astype(np.float32), y_test.astype(np.float32).reshape((-1, 1))),
        )

    @staticmethod
    def build_estimator(hyperparams, train_data, test=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract info from training data
        X, y, *_ = train_data
        in_features = X.shape[1]

        callbacks = [
            ("r2_score_valid", EpochScoring("r2", lower_is_better=False)),
            (
                "early_stopping",
                EarlyStopping(monitor="valid_loss", patience=5, lower_is_better=True),
            ),
            (
                "learning_rate_scheduler",
                LRScheduler(
                    policy=lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    # Following kargs are passed to the
                    # lr scheduler constructor
                    mode="min",
                    min_lr=1e-5,
                ),
            ),
        ]

        return NeuralNetRegressor(
            NNModule,
            criterion=nn.MSELoss,
            optimizer=torch.optim.SGD,
            max_epochs=300,
            iterator_train__shuffle=True,  # Shuffle training data on each epoch
            callbacks=callbacks,
            device=device,
            train_split=CVSplit(cv=5, random_state=RANDOM_STATE),
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            module__in_features=in_features,
            module__n_layers=hyperparams["n_layers"],
            module__n_neuron_per_layer=hyperparams["n_neuron_per_layer"],
            module__activation=getattr(F, hyperparams["activation"]),
            module__p_dropout=hyperparams["p_dropout"],
            optimizer__momentum=hyperparams["momentum"],
            optimizer__weight_decay=hyperparams["weight_decay"],
            optimizer__nesterov=True,
            verbose=3,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
        )

    hp_space = {
        "lr": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-1)),
        "batch_size": 128,
        "n_neuron_per_layer": scope.int(hp.quniform("layer_size", 10, 100, 3)),
        "activation": hp.choice("activation", ["relu", "leaky_relu", "selu"]),
        "p_dropout": hp.uniform("p_dropout", 0.0, 0.5),
        "momentum": hp.uniform("momentum", 0.87, 0.99),
        "weight_decay": hp.loguniform("alpha", np.log(1e-7), np.log(1e-2)),
        "n_layers": hp.choice("n_layers", [2, 3, 4, 5]),
    }


class NNModule(nn.Module):
    def __init__(
        self,
        in_features,
        n_layers,
        n_neuron_per_layer=10,
        activation=F.relu,
        p_dropout=0.5,
    ):
        super(NNModule, self).__init__()

        self.first_layer = Layer(in_features, n_neuron_per_layer, activation, p_dropout)
        self.middle_layers = ListModule(self, "middle_layer")
        for _ in range(n_layers - 2):
            self.middle_layers.append(
                Layer(n_neuron_per_layer, n_neuron_per_layer, activation, p_dropout)
            )
        self.fc = nn.Linear(n_neuron_per_layer, 1)

    def forward(self, X, **kwargs):
        X = self.first_layer(X)
        for layer in self.middle_layers:
            X = layer(X)
        X = self.fc(X)
        return X


class Layer(nn.Module):
    def __init__(self, in_features, out_feature, activation, p_dropout):
        super(Layer, self).__init__()
        self.dense = nn.Linear(in_features, out_feature)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(out_feature)

    def forward(self, X):
        X = self.dense(X)
        X = self.activation(X)
        X = self.dropout(X)
        return X


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))
