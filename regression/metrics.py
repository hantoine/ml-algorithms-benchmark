from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score


def compute_metric(y, y_pred, metric_name):
    if metric_name == '-rmse':
        score = mean_squared_error(y, y_pred)
    elif metric_name == 'r2':
        score = r2_score(y, y_pred)
    else:
        raise NotImplementedError(f'Metric {metric_name} not implemented')
    return score


def aggregate_metrics(metric_values, metric_name):
    if metric_name == '-rmse':
        # Each fold has approximately the same number of samples
        score = sum(metric_values) / len(metric_values)
        score = -sqrt(score)
    elif metric_name == 'r2':
        score = sum(metric_values) / len(metric_values)
    else:
        raise NotImplementedError(f'Metric {metric_name} not implemented')
    return score
