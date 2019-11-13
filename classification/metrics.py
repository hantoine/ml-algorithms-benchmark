import numpy as np

def compute_score(metric, confusion_matrix):
    is_binary_classification = confusion_matrix.shape == (2, 2)
    if is_binary_classification:
        score = compute_score_binary(metric, confusion_matrix)
    else:
        score = compute_score_multiclass(metric, confusion_matrix)
    
    return score

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
    return -score

def compute_score_multiclass(metric, confusion_matrix):
    if metric == 'accuracy':
        correct = np.sum(np.diag(confusion_matrix))
        total = np.sum(confusion_matrix)
        return -correct / total
    else:
        raise NotImplementedError(f'Metric {metric} not implemented')

