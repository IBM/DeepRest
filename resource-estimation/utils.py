import numpy as np


def sliding_window(ts, window_size):
    return np.asarray([ts[i:i+window_size] for i in range(len(ts) - window_size)])


def get_metric_with_unit(metric):
    metric_with_unit = metric
    unit = ''
    if metric == 'cpu':
        metric_with_unit = 'CPU (millicores)'
        unit = '(millicores)'
    elif metric == 'memory':
        metric_with_unit = 'Working Set Size (MB)'
        unit = '(MB)'
    elif metric == 'write-iops':
        metric_with_unit = 'Write IOps'
        unit = ''
    elif metric == 'write-tp':
        metric_with_unit = 'Write Throughput (KB)'
        unit = '(KB)'
    elif metric == 'usage':
        metric_with_unit = 'Disk Usage (MB)'
        unit = '(MB)'
    return metric_with_unit, unit
