import numpy as np
import torch
import torch.nn.functional as F
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from skorch.dataset import CVSplit
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from config import RANDOM_STATE
from utils import NonTreeBasedModel
from skorch.callbacks import Callback

class Cifar10CustomModel(NonTreeBasedModel):
    @classmethod
    def prepare_dataset(cls, train_data, test_data, categorical_features=None):
        (X_train, y_train), (X_test, y_test) = train_data, test_data
        
        return ((X_train, y_train), # normalization will be done separately for validation and training sets
                cls.DatasetWithTransforms(X_test, y_test, cls.transform_test))

    @classmethod
    def build_estimator(cls, hyperparams, train_data, test=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        early_stopping_val_percent = 10

        n_training_examples = len(train_data[0]) * (1 - (early_stopping_val_percent / 100))
        n_iter_per_epoch = n_training_examples / hyperparams['batch_size']
        n_iter_btw_restarts = int(hyperparams['epochs_btw_restarts'] * n_iter_per_epoch)
        callbacks = [
            ('fix_seed', cls.FixRandomSeed(RANDOM_STATE)),
            ('accuracy_score_valid', EpochScoring('accuracy', lower_is_better=False, on_train=True)),
            ('learning_rate_scheduler', LRScheduler(policy=CosineAnnealingWarmRestarts,
                                                    T_0=n_iter_btw_restarts,
                                                    T_mult=hyperparams['epochs_btw_restarts_mult']
                                                   ))
        ]

        def validation_split(X, y):
            splitter = CVSplit(cv=int(100 / early_stopping_val_percent), random_state=RANDOM_STATE)
            dataset_train, dataset_valid = splitter(X)
            return (cls.TransformsDatasetWrapper(dataset_train, cls.transform_train),
                    cls.TransformsDatasetWrapper(dataset_valid, cls.transform_test))

        return NeuralNetClassifier(
            cls.CifarCustomNet,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.SGD,
            max_epochs=40 if not test else 1,
            iterator_train__shuffle=True, # Shuffle training data on each epoch
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            callbacks=callbacks,
            device=device,
            train_split=validation_split,
            lr=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            optimizer__momentum=hyperparams['momentum'],
            optimizer__weight_decay=hyperparams['weight_decay'],
            optimizer__nesterov=hyperparams['nesterov'],
            module__conv_dropout=hyperparams['conv_dropout'],
            module__fc_dropout=hyperparams['fc_dropout'],
            verbose=3
        )

    hp_space = {
        'batch_size': 64,
        'learning_rate': 1.6e-2,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'nesterov': True,
        'conv_dropout': 0.1,
        'fc_dropout': 0.15,
        'epoch_btw_restarts': 10,
        'epochs_btw_restarts_mult': 1.0
    }

    class CifarCustomNet(nn.Module):
        def __init__(self, conv_dropout, fc_dropout):
            super(Cifar10CustomModel.CifarCustomNet, self).__init__()
            config_map = {
                1: [
                    [3, 32]
                ],
                2: [
                    [3, 32],
                    [32, 64]
                ],
                3: [
                    [3, 16],
                    [16, 32],
                    [32, 64]
                ],
                4: [
                    [32 + 64 + 64, 240],
                    [240 * 5 * 5, 1024],
                    [1024, 512],
                    [512, 128],
                    [128, 10]
                ]
            }
            
            # basis
            config_maps1 = config_map[1]
            self.layer1 = nn.Sequential(nn.Conv2d(config_maps1[0][0], config_maps1[0][1], 5),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout),
                                        nn.MaxPool2d(2, 2)) 
            # higher resolution
            config_maps2 = config_map[2]
            self.layer121 = nn.Sequential(nn.Conv2d(config_maps2[0][0], config_maps2[0][1], 3),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout))

            self.layer122 = nn.Sequential(nn.Conv2d(config_maps2[1][0], config_maps2[1][1], 3),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout),
                                        nn.MaxPool2d(2, 2))
            # smaller resolution
            config_maps3 = config_map[3]
            self.layer131 = nn.Sequential(nn.Conv2d(config_maps3[0][0], config_maps3[0][1], 7),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout),
                                        nn.MaxPool2d(2, 1))
            self.layer132 = nn.Sequential(nn.Conv2d(config_maps3[1][0], config_maps3[1][1], 7),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout),
                                        nn.MaxPool2d(2, 1))
            self.layer133 = nn.Sequential(nn.Conv2d(config_maps3[2][0], config_maps3[2][1], 5), 
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout))
            
            # final combination
            config_maps4 = config_map[4]
            self.layer2 = nn.Sequential(nn.Conv2d(config_maps4[0][0], config_maps4[0][1], 5),
                                        nn.ReLU(),
                                        nn.Dropout2d(conv_dropout),
                                        nn.MaxPool2d(2, 2))
            self.layer3 = nn.Sequential(nn.Linear(config_maps4[1][0], config_maps4[1][1]),
                                        nn.ReLU(),
                                        nn.Dropout(fc_dropout))            
            self.layer4 = nn.Sequential(nn.Linear(config_maps4[2][0], config_maps4[2][1]),
                                        nn.ReLU(),
                                        nn.Dropout(fc_dropout))
            self.layer5 = nn.Sequential(nn.Linear(config_maps4[3][0], config_maps4[3][1]),
                                        nn.ReLU(),
                                        nn.Dropout(fc_dropout))
            self.layer6 = nn.Linear(config_maps4[4][0], config_maps4[4][1])

            return

        def forward(self, img):
            x = self.layer1(img)

            y = self.layer121(img)
            y = self.layer122(y)

            z = self.layer131(img)
            z = self.layer132(z)
            z = self.layer133(z)

            x = torch.cat((x, y, z), 1)
            x = self.layer2(x)
            
            x = x.view(-1, 240 * 5 * 5)

            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)

            return x

    class DatasetWithTransforms(Dataset):
        def __init__(self, X, y, transforms=None):
            assert len(X) == len(y)
            self.X = X
            self.y = y
            self.transforms = transforms

        def __getitem__(self, index):
            X = self.X[index]
            y = self.y[index]

            img = Image.fromarray(255 * (1 - np.moveaxis(X.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])))
            return self.transforms(img), y

        def __len__(self):
            return len(self.X)
        
        def plot(self, index):
            X = self.X[index]
            img = np.moveaxis(X.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])
            plt.imshow(img)

    class TransformsDatasetWrapper(Dataset):
        def __init__(self, dataset, transforms):
            self.dataset = dataset
            self.transforms = transforms

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            X, y = self.dataset[index]
            img = Image.fromarray(255 * (1 - np.moveaxis(X.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])))
            return self.transforms(img), y


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.48216, 0.4465), (0.247, 0.24346, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.48216, 0.4465), (0.247, 0.24346, 0.2616))
    ])

    class FixRandomSeed(Callback):  
        def __init__(self, seed=42):
            self.seed = seed
        
        def initialize(self):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            
            try:
                random.seed(self.seed)
            except NameError:
                import random
                random.seed(self.seed)

            np.random.seed(self.seed)
            torch.backends.cudnn.deterministic=True
