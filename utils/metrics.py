import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix as compute_confusion_matrix

def compute_metric(y, y_pred, metric_name):
    if metric_name == 'accuracy' or metric_name == 'f1':
        score = compute_confusion_matrix(y, y_pred)
    elif metric_name == 'rmse':
        score = mean_squared_error(y, y_pred)
    elif metric_name == 'r2':
        score = r2_score(y, y_pred)
    else:
        raise NotImplementedError(f'Metric {metric_name} not implemented')
    return score


def compute_loss(metric_name, metric_values):
    if metric_name == 'accuracy' or metric_name == 'f1':
        confusion_matrices = metric_values
        confusion_matrix = np.array(confusion_matrices).sum(axis=0)
        is_binary_classification = confusion_matrix.shape == (2, 2)
        if is_binary_classification:
            score = compute_score_binary(metric_name, confusion_matrix)
        else:
            score = compute_score_multiclass(metric_name, confusion_matrix)
    elif metric_name == 'rmse':
        # Each fold has approximately the same number of samples
        score = sum(metric_values) / len(metric_values)
        score = -sqrt(score)
    elif metric_name == 'r2':
        score = sum(metric_values) / len(metric_values)
    else:
        raise NotImplementedError(f'Metric {metric_name} not implemented')
    return -score

def compute_score_binary(metric, confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    if metric == 'f1':
        # Averaging of f1 across folds as suggested by Forman & Scholz
        # https://www.hpl.hp.com/techreports/2009/HPL-2009-359.pdf
        score = (2*tp) / (2*tp + fp + fn)
    elif metric == 'accuracy':
        score = (tp + tn) / (tn + fp + fn + tp)
    else:
        raise NotImplementedError(f'Metric {metric} not implemented')
    return score


def compute_score_multiclass(metric, confusion_matrix):
    if metric == 'accuracy':
        correct = np.sum(np.diag(confusion_matrix))
        total = np.sum(confusion_matrix)
        score = correct / total
    elif metric == 'f1':
        f1_per_class = np.empty(len(confusion_matrix))
        for class_idx in range(len(confusion_matrix)):
            tp = confusion_matrix[class_idx, class_idx]
            fp = np.sum(confusion_matrix[:, class_idx]) - tp
            fn = np.sum(confusion_matrix[class_idx, :]) - tp
            f1 = (2*tp) / (2*tp + fp + fn)
            f1_per_class[class_idx] = f1
        score = f1_per_class.mean()
    else:
        raise NotImplementedError(f'Metric {metric} not implemented')
    return score
