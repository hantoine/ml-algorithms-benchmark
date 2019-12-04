import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from hyperopt import hp
from hyperopt.pyll import scope
from PIL import Image
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import CVSplit, Dataset as SkorchDataset
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, Subset

from config import RANDOM_STATE
from utils import NonTreeBasedModel


class Cifar10CustomModel(NonTreeBasedModel):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def prepare_dataset(cls, train_data, test_data, categorical_features=None):
        return train_data, test_data #Cifar10 is already clean

    @classmethod
    def build_estimator(cls, hyperparams, train_data, verbose=True, test=False): #  change default verbose to false later
        early_stopping_val_percent = 10

        n_training_examples = len(train_data[0]) * (1 - (early_stopping_val_percent / 100))
        n_iter_per_epoch = n_training_examples / hyperparams['batch_size']
        n_iter_btw_restarts = int(hyperparams['epochs_btw_restarts'] * n_iter_per_epoch)
        callbacks = [
            ('fix_seed', cls.FixRandomSeed(RANDOM_STATE)),
            ('lr_monitor', cls.LRMonitor()),
            ('accuracy_score_valid', EpochScoring('accuracy', lower_is_better=False, on_train=True)),
            ('early_stopping', EarlyStopping(monitor='valid_acc', lower_is_better=False, patience=5)),
            ('learning_rate_scheduler', LRScheduler(policy=cls.SkorchCosineAnnealingWarmRestarts,
                                                    T_0=n_iter_btw_restarts,
                                                    T_mult=hyperparams['epochs_btw_restarts_mult']
                                                   ))
        ]

        def validation_split(X, y):
            """ Custom split is used to apply augmentation to the training set only """
            splitter = CVSplit(cv=int(100 / early_stopping_val_percent), random_state=RANDOM_STATE)
            dataset_train, dataset_valid = splitter(X)
            dataset_train = cls.AugmentedDataset(dataset_train)
            return dataset_train, dataset_valid

        return NeuralNetClassifier(
            cls.CifarCustomNet,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.SGD,
            max_epochs=hyperparams['max_epochs'] if not test else 1,
            iterator_train__shuffle=True,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            dataset=cls.NormalizedDataset,
            callbacks=callbacks,
            device=cls.device,
            train_split=validation_split,
            lr=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            optimizer__momentum=hyperparams['momentum'],
            optimizer__weight_decay=hyperparams['weight_decay'],
            optimizer__nesterov=hyperparams['nesterov'],
            module__conv_dropout=hyperparams['conv_dropout'],
            module__fc_dropout=hyperparams['fc_dropout'],
            verbose=3 if verbose else 0
        )

    hp_space = {
        'batch_size': 64,
        'learning_rate': 2e-2,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'nesterov': True,
        'conv_dropout': 0.15,
        'fc_dropout': 0.15,
        'epochs_btw_restarts': 41,
        'epochs_btw_restarts_mult': 2,
        'max_epochs': 30
    }

    class CifarCustomNet(nn.Module):
        def __init__(self, conv_dropout, fc_dropout):
            super(Cifar10CustomModel.CifarCustomNet, self).__init__()
            config = {
                'branch1': [
                    [3, 32]
                ],
                'branch2': [
                    [3, 64],
                    [64, 128]
                ],
                'branch3': [
                    [3, 32],
                    [32, 64],
                    [64, 64]
                ],
                'head': [
                    [32 + 128 + 64, 256],
                    [256, 256],
                    [256, 10]
                ]
            }
            
            # basis
            self.branch1 = nn.Sequential(nn.Conv2d(*config['branch1'][0], 5),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(config['branch1'][0][1]),
                                         nn.Dropout2d(conv_dropout),
                                         nn.MaxPool2d(2, 2))
            # higher resolution
            self.branch2_layer1 = nn.Sequential(nn.Conv2d(*config['branch2'][0], 3),
                                                nn.ReLU(),
                                                nn.Dropout2d(conv_dropout))

            self.branch2_layer2 = nn.Sequential(nn.Conv2d(*config['branch2'][1], 3),
                                                nn.ReLU(),
                                                nn.BatchNorm2d(config['branch2'][1][1]),
                                                nn.Dropout2d(conv_dropout),
                                                nn.MaxPool2d(2, 2))
            # smaller resolution
            self.branch3_layer1 = nn.Sequential(nn.Conv2d(*config['branch3'][0], 7),
                                                nn.ReLU(),
                                                nn.Dropout2d(conv_dropout),
                                                nn.MaxPool2d(2, 1))
            self.branch3_layer2 = nn.Sequential(nn.Conv2d(*config['branch3'][1], 7),
                                                nn.ReLU(),
                                                nn.Dropout2d(conv_dropout),
                                                nn.MaxPool2d(2, 1))
            self.branch3_layer3 = nn.Sequential(nn.Conv2d(*config['branch3'][2], 5),
                                                nn.BatchNorm2d(config['branch3'][2][1]),
                                                nn.ReLU(),
                                                nn.Dropout2d(conv_dropout))
            
            # head
            self.head_layer1 = nn.Sequential(nn.Conv2d(*config['head'][0], 3, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(config['head'][0][1]),
                                             nn.Dropout2d(conv_dropout),
                                             )
            self.head_layer2 = nn.Sequential(nn.Conv2d(*config['head'][1], 5),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(config['head'][1][1]),
                                             nn.Dropout2d(conv_dropout),
                                             )
            self.fc = nn.Linear(*config['head'][2])

            return

        def forward(self, img):
            x = self.branch1(img)

            y = self.branch2_layer1(img)
            y = self.branch2_layer2(y)

            z = self.branch3_layer1(img)
            z = self.branch3_layer2(z)
            z = self.branch3_layer3(z)

            x = torch.cat((x, y, z), 1)
            x = self.head_layer1(x)
            
            # Global Average Pooling
            # import pdb; pdb.set_trace()
            x = F.avg_pool2d(x, x.shape[-1]).view(-1, x.shape[1])

            x = self.fc(x)

            return x

    class AugmentedDataset(Dataset):
        """ Used in validation_split function to apply augmentations on training set """
        def __init__(self, dataset_subset):
            self.dataset_subset = dataset_subset

            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.48216, 0.4465), (0.247, 0.24346, 0.2616))
            ])

        def __len__(self):
            return len(self.dataset_subset)

        def __getitem__(self, index):
            original_index = self.dataset_subset.indices[index]
            img, y = self.dataset_subset.dataset.get_img(original_index)
            return self.transforms(img), y

    class NormalizedDataset(SkorchDataset):
        """ Used to reshape examples and lazily normalize the dataset """
        def __init__(self, X, y=None):
            super(Cifar10CustomModel.NormalizedDataset, self).__init__(X, y)
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.48216, 0.4465), (0.247, 0.24346, 0.2616))
            ])

        def __getitem__(self, index):
            img, y = self.get_img(index)
            return self.transforms(img), y

        def get_img(self, index):
            """ Used by AugmentedDataset to get non-transformed PIL image """
            img, y = self.get_raw(index)
            img = Image.fromarray(255 * (1 - img))
            return img, y

        def get_raw(self, index):
            " Useful to plot an image "
            X, y = super(Cifar10CustomModel.NormalizedDataset, self).__getitem__(index)
            img = np.moveaxis(X.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])
            return img, y

    class FixRandomSeed(Callback):
        """ Skorch callback used to fix seeds at the beginning of the training """
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

    class LRMonitor(Callback):
        """ Skorch callback used to save the learning rate every epoch """
        def on_epoch_end(self, net, **kwargs):
            net.history.record('lr', net.optimizer_.param_groups[0]['lr'])

    class SkorchCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
        """
            CosineAnnealingWarmRestarts scheduler with additional '.batch_step()' method
            called by Skorch every batch. This class is necessary to ensure the learning rate is
            updated every batch as opposed to every epoch (Skorch default behavior)
        """
        def batch_step(self, batch_idx):
            super(Cifar10CustomModel.SkorchCosineAnnealingWarmRestarts, self).step(batch_idx)
