from os.path import isfile
from os.path import join as joinpath

import matplotlib.pyplot as plt
import torch
from skimage.transform import resize

from classifier_interpretability import datasets, models
from config import RESULTS_DIR
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
    cifar_model_weights_path = joinpath(RESULTS_DIR, 'Cifar10CustomModel-weights.pkl')
    if not isfile(cifar_model_weights_path):
        raise ValueError('Cifar10 CNN weights not saved, run training first')

    model = models.Cifar10CustomModel
    dataset = datasets.Cifar10Dataset

    train, test = dataset.get()
    train_data, test_data = \
        model.prepare_dataset(train, test, dataset.categorical_features)
    tuning_results = get_tuning_results(dataset, model)
    estimator = model.build_estimator(tuning_results['hp'], train_data)
    
    estimator.initialize()
    estimator.load_params(f_params=cifar_model_weights_path)

    # CMAP
    module = estimator.module_
    module.eval()
    def save_final_conv_output(module, inputs, outputs):
        save_final_conv_output.output = outputs.data
    module.head_layer2.register_forward_hook(save_final_conv_output)

    test_dataset = model.NormalizedDataset(*test_data)
    x, y = test_dataset[image_index]

    x = x.unsqueeze(dim=0).to(model.device)
    y_pred = module(x)
    y_pred = torch.argmax(y_pred)

    last_conv_output = save_final_conv_output.output.reshape(240, -1)
    pred_class_weights = module.fc._parameters['weight'].data[y_pred]
    cmap = pred_class_weights.matmul(last_conv_output)
    cmap -= cmap.min()
    cmap /= cmap.max()
    height, width = save_final_conv_output.output.shape[-2:]
    cmap = cmap.reshape(height, width).cpu().numpy()

    if y == y_pred:
        print(f'The model correctly identify {dataset.classes[y]}')
    else:
        print(f'The model failed to identify a {dataset.classes[y]}, it predicted {dataset.classes[y_pred]}')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(test_dataset.get_raw(image_index)[0])
    plt.subplot(1, 3, 2)
    plt.imshow(resize(cmap, (32, 32)), cmap='jet')
    plt.subplot(1, 3, 3)
    plt.imshow(test_dataset.get_raw(image_index)[0])
    plt.imshow(resize(cmap, (32, 32)), alpha=0.3, cmap='jet')
    plt.show()

