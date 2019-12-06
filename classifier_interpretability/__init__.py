from os.path import isfile
from os.path import join as joinpath

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from sklearn.tree import export_graphviz, plot_tree
from torchvision import transforms

from classifier_interpretability import datasets, models
from config import RESULTS_DIR, RANDOM_STATE
from utils import (train_all_models_on_all_datasets,
                   tune_all_models_on_all_datasets)
from utils.training import get_tuning_results


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
    show_decision_tree_visualizations()

    estimator, test_dataset = prepare_model()
    generate_class_activation_maps(estimator, test_dataset, image_index)
    generate_activation_mazimization_viz(estimator, test_dataset)

def generate_activation_mazimization_viz(estimator, test_dataset):
    plt.figure()
    plt.subplots_adjust(wspace=0.35, hspace=0)
    plt.subplot(2, 3, 1)
    loss_values1 = plot_activation_maximization(estimator, test_dataset, class_vizualized=0, n_epochs=80,
                                                lr=3e-2, momentum=0.9, type_initial_x='dataset',
                                                image_index=10)
    plt.subplot(2, 3, 2)
    loss_values2 = plot_activation_maximization(estimator, test_dataset, class_vizualized=0, n_epochs=160,
                                                lr=1e-2, momentum=0.9, type_initial_x='zero',
                                                image_index=10)
    plt.subplot(2, 3, 3)
    loss_values3 = plot_activation_maximization(estimator, test_dataset, class_vizualized=0, n_epochs=100,
                                                lr=1e-3, momentum=0.9, type_initial_x='random',
                                                image_index=10)
    plt.subplot(4, 3, 7)
    plt.plot(loss_values1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.subplot(4, 3, 8)
    plt.plot(loss_values2)
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.subplot(4, 3, 9)
    plt.plot(loss_values3)
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.show()


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


def show_decision_tree_visualizations():
    model = models.DecisionTreeModel
    dataset = datasets.Cifar10Dataset

    train, test = dataset.get()
    train_data, test_data = model.prepare_dataset(train, test, dataset.categorical_features)

    tuning_results = get_tuning_results(dataset, model)
    estimator = model.build_estimator(tuning_results['hp'], train_data)
    estimator.fit(*train_data)

    X = np.arange(3072)
    feature_names = np.empty(3072, dtype=np.dtype('U7'))
    for i, row in enumerate(np.moveaxis(X.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])):
        for j, pixel in enumerate(row):
            for index, color in zip(pixel, 'RGB'):
                feature_names[index] = f'{color}-{i}-{j}'

    show_decision_tree_visualization(estimator, dataset, feature_names, 3, rotate=False)


def show_decision_tree_visualization(estimator, dataset, feature_names, max_depth, rotate=False):
    dot_data = export_graphviz(estimator,
                           max_depth=max_depth,
                           label='none',
                           class_names=dataset.classes,
                           proportion=True,
                           impurity=False,
                           precision=0,
                           rotate=rotate,
                           rounded=True,
                           feature_names=feature_names)
    # Remove class proportions which take too much space
    dot_data = dot_data.replace('[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\\n', '')
    dot_data = dot_data.replace('[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\\n', '')
    dot_data = dot_data.replace('[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]\\n', '')

    graph = graphviz.Source(dot_data) 
    graph.view(f'decision_tree_interpretation-{max_depth}-{rotate}')


def generate_class_activation_maps(estimator, test_dataset, image_index):
    module = estimator.module_
    module.eval()
    def save_final_conv_output(module, inputs, outputs):
        save_final_conv_output.output = outputs.data
    module.head_layer2.register_forward_hook(save_final_conv_output)

    if image_index == -1:
        while True:
            command = input('Enter index of image in test set to perform class activation mapping on (q to leave): ')
            if command == 'q':
                break
            command = command.split(' ')
            image_index = int(command[0])
            show_class_activation_map(module, test_dataset, save_final_conv_output, image_index, transparency=float(command[1]) if len(command) > 1 else None)
        return

    plt.figure()
    plt.subplot(1, 3, 1)
    show_class_activation_map(module, test_dataset, save_final_conv_output, 10, transparency=0.3) # plane
    plt.subplot(1, 3, 2)
    show_class_activation_map(module, test_dataset, save_final_conv_output, 897, transparency=0.3) # horse, 48 good too
    plt.subplot(1, 3, 3)
    show_class_activation_map(module, test_dataset, save_final_conv_output, 91, transparency=0.3) # cat
    plt.show()


def show_class_activation_map(module, test_dataset, hook, image_index, transparency):
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

    if transparency == None:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(test_dataset.get_img(image_index)[0])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 3, 2)
        plt.imshow(test_dataset.get_img(image_index)[0])
        plt.imshow(resize(cmap, (32, 32)), alpha=0.35, cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 3, 3)
        plt.imshow(resize(cmap, (32, 32)), cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        plt.imshow(test_dataset.get_img(image_index)[0])
        plt.imshow(resize(cmap, (32, 32)), alpha=transparency, cmap='jet')
        plt.xticks([])
        plt.yticks([])

def plot_activation_maximization(estimator, test_dataset, class_vizualized=0, n_epochs=40,
                                 lr=3e-2, momentum=0.9, type_initial_x='dataset',
                                 image_index=10):
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic=True

    module = estimator.module_
    module.eval()

    if type_initial_x == 'zero':
        x = torch.zeros((3, 32, 32))
    elif type_initial_x == 'dataset':
        x, y = test_dataset[image_index]
    elif type_initial_x == 'random':
        x = torch.zeros((3, 32, 32))
        x.normal_(mean=0, std=0.2)
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
    plt.imshow(max_activation)
    plt.xticks([])
    plt.yticks([])
    return loss_values
