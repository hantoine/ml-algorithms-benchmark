from os.path import isfile
from os.path import join as joinpath

import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
import numpy as np

from classifier_interpretability import datasets, models
from config import RESULTS_DIR
from utils import (train_all_models_on_all_datasets,
                   tune_all_models_on_all_datasets)
from utils.training import get_tuning_results
from torchvision import transforms


def tune_all_models_on_all_clf_interpret_datasets(tuning_trials_per_step=5, max_tuning_time=120,
                                                                max_trials_without_improvement=150,
                                                                tuning_step_max_time=60,
                                                                mongo_address=None):
    tune_all_models_on_all_datasets('classification', datasets.all_datasets, models.all_models,
                                    tuning_trials_per_step, max_tuning_time,
                                    max_trials_without_improvement, tuning_step_max_time,
                                    mongo_address)


def evaluate_all_models_on_all_clf_interpret_datasets(max_training_time=180):
    train_all_models_on_all_datasets(datasets.all_datasets, models.all_models, max_training_time)


def generate_interpretation_viz(image_index):
    estimator, test_dataset = prepare_model()

    # CAM
    generate_class_activation_maps(estimator, test_dataset, image_index)

    generate_activation_maximization_viz(estimator, test_dataset, class_vizualized=0, n_epochs=80,
                                         lr=3e-2, momentum=0.9, type_initial_x='dataset',
                                         image_index=10)
    generate_activation_maximization_viz(estimator, test_dataset, class_vizualized=1, n_epochs=80,
                                        lr=3e-2, momentum=0.9, type_initial_x='dataset',
                                        image_index=10)
    generate_activation_maximization_viz(estimator, test_dataset, class_vizualized=0, n_epochs=80,
                                        lr=3e-2, momentum=0.9, type_initial_x='random',
                                        image_index=10)


def prepare_model():
    cifar_model_weights_path = joinpath(RESULTS_DIR, 'Cifar10CustomModel-weights.pkl')
    if not isfile(cifar_model_weights_path):
        raise ValueError('Cifar10 CNN weights not saved, run training first')

    model = models.Cifar10CustomModel
    dataset = datasets.Cifar10Dataset

    train, test = dataset.get()
    train_data, test_data = model.prepare_dataset(train, test, dataset.categorical_features)
    tuning_results = get_tuning_results(dataset, model)
    estimator = model.build_estimator(tuning_results['hp'], train_data)

    estimator.initialize()
    estimator.load_params(f_params=cifar_model_weights_path)

    test_dataset = model.NormalizedDataset(*test_data)

    return estimator, test_dataset


def generate_class_activation_maps(estimator, test_dataset, image_index):
    module = estimator.module_
    module.eval()
    def save_final_conv_output(module, inputs, outputs):
        save_final_conv_output.output = outputs.data
    module.head_layer2.register_forward_hook(save_final_conv_output)

    while True:
        command = input('Enter index of image in test set to perform class activation mapping on (q to leave): ')
        if command == 'q':
            break
        image_index = int(command)
        show_class_activation_map(module, test_dataset, save_final_conv_output, image_index)


def show_class_activation_map(module, test_dataset, hook, image_index):
    model = models.Cifar10CustomModel
    dataset = datasets.Cifar10Dataset
    x, y = test_dataset[image_index]

    x = x.unsqueeze(dim=0).to(model.device)
    y_pred = module(x)
    y_pred = torch.argmax(y_pred)

    n_feature_map = hook.output.shape[1]
    last_conv_output = hook.output.reshape(n_feature_map, -1)
    pred_class_weights = module.fc._parameters['weight'].data[y_pred]
    cmap = pred_class_weights.matmul(last_conv_output)
    cmap -= cmap.min()
    cmap /= cmap.max()
    height, width = hook.output.shape[-2:]
    cmap = cmap.reshape(height, width).cpu().numpy()

    if y == y_pred:
        print(f'The model correctly identify {dataset.classes[y]}')
    else:
        print(f'The model failed to identify a {dataset.classes[y]}, it predicted {dataset.classes[y_pred]}')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(test_dataset.get_img(image_index)[0])
    plt.subplot(1, 3, 2)
    plt.imshow(test_dataset.get_img(image_index)[0])
    plt.imshow(resize(cmap, (32, 32)), alpha=0.35, cmap='jet')
    plt.subplot(1, 3, 3)
    plt.imshow(resize(cmap, (32, 32)), cmap='jet')
    plt.title(f'Class activation mapping for image')# {image_index}')
    plt.show()

def generate_activation_maximization_viz(estimator, test_dataset, class_vizualized=0, n_epochs=40,
                                         lr=3e-2, momentum=0.9, type_initial_x='dataset',
                                         image_index=10):
    module = estimator.module_
    module.eval()

    if type_initial_x == 'zero':
        x = torch.zeros((3, 32, 32))
    elif type_initial_x == 'dataset':
        x, y = test_dataset[image_index]
    elif type_initial_x == 'random':
        x = torch.zeros((3, 32, 32))
        x.normal_(mean=0, std=0.3)
    x = x.unsqueeze(dim=0).to(models.Cifar10CustomModel.device)
    x.requires_grad_(True)

    y = torch.tensor([class_vizualized], dtype=torch.long,
                 device=models.Cifar10CustomModel.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr, momentum)

    loss_values = []
    for _ in range(n_epochs):
        y_pred = module(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    print(f'Final loss value: {loss_values[-1]}')
    max_activation = x.data.squeeze()


    denormalize = transforms.Compose([
        transforms.Normalize((0.0, 0.0, 0.0), (1 / 0.247, 1 / 0.24346, 1 / 0.2616)),
        transforms.Normalize((-0.4914, -0.48216, -0.4465), (1.0, 1.0, 1.0))
    ])

    max_activation = denormalize(max_activation).cpu().numpy()
    max_activation = np.moveaxis(max_activation, [0, 1, 2], [2, 0, 1])
    if type_initial_x == 'dataset':
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.title('Activation maximization learning curve')
        plt.subplot(2, 2, 3)
        plt.imshow(test_dataset.get_img(image_index)[0])
        plt.subplot(2, 2, 4)
        plt.imshow(max_activation)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.title('Activation maximization learning curve')
        plt.subplot(2, 1, 2)
        plt.imshow(max_activation)
        plt.tight_layout()
        plt.show()