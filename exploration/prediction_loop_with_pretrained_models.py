from baselines import Baseline_Resource_Only, Baseline_RequestAndResource
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import copy
plt.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


window_seconds = 300
sampling_interval = 5
step_size = window_seconds // sampling_interval
dataset = 'composePost_uploadMedia_readUserTimeline-ransomware'
num_layers = 1
hidden_layer_size = 128
num_epochs = 50
quantiles = (.05, .50, .95)
batch_size = 32
learning_rate = 0.001

# ================================================== compute ==================================================
# media-frontend        | media-mongodb         | nginx-thrift
# compose-post-service  |
# user-timeline-service | user-timeline-mongodb |
# post-storage-service  | post-storage-mongodb  |
# =============================================================================================================
# available metrics: cpu | memory
# =================================== storage ====================================
# media-mongodb-pvc       | post-storage-mongodb-pvc | user-timeline-mongodb-pvc |
# ================================================================================
# available metrics: write-iops | write-tp | usage

component2metrics = {
    'media-frontend': ['cpu', 'memory'],
    'media-mongodb': ['cpu', 'memory'],
    'nginx-thrift': ['cpu', 'memory'],
    'compose-post-service': ['cpu', 'memory'],
    'user-timeline-service': ['cpu', 'memory'],
    'user-timeline-mongodb': ['cpu', 'memory'],
    'post-storage-service': ['cpu', 'memory'],
    'post-storage-mongodb': ['cpu', 'memory'],
    'media-mongodb-pvc': ['write-iops', 'write-tp', 'usage'],
    'post-storage-mongodb-pvc': ['write-iops', 'write-tp', 'usage'],
    'user-timeline-mongodb-pvc': ['write-iops', 'write-tp', 'usage'],

}


########################################################################################################################
def load_data(dataset, component, metric):
    with open('./data/%s.pkl' % dataset, 'rb') as f:
        X, y, y_names, meta = pickle.load(f)
    for i, y_name in enumerate(y_names):
        for k, v in (('cpu', 1e3),          # cores -> millicores
                     ('memory', 1e-6),      # bytes -> megabytes
                     ('tp', (512 * 1e-3)),  # blocks -> kilobytes
                     ('usage', 1e3)):       # gigabytes -> megabytes
            if k in y_name:
                y[:, [i], :] *= v
    y = y[:, [y_names.index(i + '_%s' % metric) for i in [component]], :]
    return X, y.swapaxes(1, 2), meta


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


def ts_IOU(ts1, ts2):
    """
    This function computes the percentage similarity (measured in intersection-over-union) between two input
    time-series, ts1 and ts2.
    :param ts1: One of the time-series to be compared
    :param ts2: One of the time-series to be compared
    :return: similarity between two input time series
    """
    assert len(ts1) == len(ts2), 'The lengths of two time-series do not match (%d != %d)' % (len(ts1), len(ts2))
    intersection, union = 0, 0
    for i in range(len(ts1)):
        y1, y2 = ts1[i], ts2[i]
        if y1 >= 0 and y2 >= 0:
            intersection += min(y1, y2)
            union += max(y1, y2)
        elif y1 < 0 and y2 < 0:
            intersection += abs(max(y1, y2))
            union += abs(min(y1, y2))
        else:
            union += (abs(y1) + abs(y2))
    return intersection / union


class QuantileRNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, widths, dropout=0.50, bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.widths = widths
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size, hidden_layer_size, num_layers=num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(hidden_layer_size * (2 if bidirectional else 1), output_size)

        self.dummy = nn.Parameter(torch.full((1,), fill_value=1.), requires_grad=False)
        self.cumsum_linear_1 = nn.Linear(1, hidden_layer_size)
        self.cumsum_linear_2 = nn.Linear(hidden_layer_size, 2)

        self.prob_linears = nn.ModuleList([nn.ModuleList([nn.Linear(1, hidden_layer_size),
                                                          nn.Linear(hidden_layer_size, width)]) for width in widths])

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        prob_cumsum = self.softmax(self.cumsum_linear_2(self.relu(self.cumsum_linear_1(self.dummy))))
        prob_cumsum = torch.cat([prob_cumsum[[0]].repeat(self.widths[0] + self.widths[1]),
                                 prob_cumsum[[1]].repeat(self.widths[0] + self.widths[1])])

        prob = torch.cat([self.softmax(l2(self.relu(l1(self.dummy)))) for l1, l2 in self.prob_linears])

        input_seq = input_seq * prob[None, None, :] * prob_cumsum[None, None, :]
        input_seq = input_seq.permute(1, 0, 2)

        hidden_cell = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size,
                                  self.hidden_layer_size).to(device)
        rnn_out, hidden_cell = self.rnn(input_seq, hidden_cell)
        rnn_out = rnn_out.permute(1, 0, 2)
        rnn_out = self.dropout(rnn_out)

        predictions = self.linear(rnn_out)
        return predictions

    @staticmethod
    def quantile_loss(outputs, labels):
        losses = []
        for i, q in enumerate(quantiles):
            errors = labels - outputs[:, :, [i]]
            losses.append(torch.max((q - 1) * errors, q * errors))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))
        return loss

    @staticmethod
    def normalization_minmax(M, split):
        min_val = np.min(M[:split])
        max_val = np.max(M[:split])
        if (max_val - min_val) != 0.0:
            M = (M - min_val) / (max_val - min_val)
        return M, min_val, max_val


output_pkl = {}
for component, metrics in component2metrics.items():
    for metric in metrics:
        ################################################################################################################
        load_path = './%s-%s_%s.pth' % (dataset, component, metric)
        offset_estimation = metric in ['memory', 'usage']
        metric_with_unit, unit = get_metric_with_unit(metric)
        timeline = [t * sampling_interval for t in range(0, step_size)]
        ################################################################################################################
        print('Loading dataset...')
        print('   > path     : %s' % dataset)
        print('   > component: %s' % component)
        print('   > metric   : %s' % metric)
        X, y, meta = load_data(dataset, component, metric)
        split = meta['split']

        print('Max-min normalization on inputs (requests)...')
        X_cumsum = np.cumsum(X, axis=1)

        X_prefix = meta['X_prefix']
        X_prefix_cumsum = np.cumsum(X_prefix, axis=1)

        X, scale_min_X, scale_max_X = QuantileRNN.normalization_minmax(X, split=split)
        scale_range_X = scale_max_X - scale_min_X
        X_cumsum, _, _ = QuantileRNN.normalization_minmax(X_cumsum, split=split)

        X_prefix, _, _ = QuantileRNN.normalization_minmax(X_prefix, split=split)
        X_prefix_cumsum, _, _ = QuantileRNN.normalization_minmax(X_prefix_cumsum, split=split)

        Xs = [X, X_prefix, X_cumsum, X_prefix_cumsum]
        X = np.concatenate(Xs, axis=2)
        widths = [x.shape[-1] for x in Xs]
        print('   > X.shape: %s' % str(X.shape))
        print('   > X.min  : %.4f' % scale_min_X)
        print('   > X.max  : %.4f' % scale_max_X)

        y_offset = np.zeros(shape=(y.shape[0], 1, 1))
        if offset_estimation:
            print('Enabling offset estimation for metric = %s...' % metric)
            y_offset = y[:, [0], :]
            y = y - y_offset

        print('Max-min normalization on outputs (resources)...')
        y, scale_min_y, scale_max_y = QuantileRNN.normalization_minmax(y, split=split)
        scale_range_y = scale_max_y - scale_min_y
        print('   > y.shape: %s' % str(y.shape))
        print('   > y.min  : %.4f' % scale_min_y)
        print('   > y.max  : %.4f' % scale_max_y)

        print('Splitting train-test datasets...')
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        y_test_offset = y_offset[split:]
        print('   > [Train             ] X: %s | y: %s' % (str(X_train.shape), str(y_train.shape)))
        print('   > [Test              ] X: %s | y: %s' % (str(X_test.shape), str(y_test.shape)))

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        model = QuantileRNN(input_size=X.shape[-1], hidden_layer_size=hidden_layer_size,
                            output_size=len(quantiles), num_layers=num_layers, widths=widths).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.load_state_dict(torch.load(load_path))
        model.eval()

        out_outputs = []
        out_labels = []
        p = []
        for iv, (inputs, labels) in enumerate(train_loader):
            if iv % step_size != 0 or iv < 150:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            labels = labels.detach().cpu().numpy().squeeze()
            outputs = outputs.detach().cpu().numpy().squeeze()

            labels = labels * scale_range_y + scale_min_y
            outputs = outputs * scale_range_y + scale_min_y

            if offset_estimation:
                labels = labels + y_offset[iv].squeeze()
                outputs = outputs + y_offset[iv].squeeze()
            outputs = np.maximum(outputs, 1e-6)

            print(np.corrcoef(outputs[:, 1], labels)[0, 1])

            out_outputs.append(outputs)
            out_labels.append(labels)
        out_outputs = np.asarray(out_outputs)
        out_outputs = np.reshape(out_outputs, (out_outputs.shape[0] * out_outputs.shape[1], out_outputs.shape[2]))[150:]
        out_labels = np.asarray(out_labels)
        out_labels = np.reshape(out_labels, (out_labels.shape[0] * out_labels.shape[1]))[150:]

        minv = min(list(out_labels) + list(out_outputs[:, 1]))
        maxv = max(list(out_labels) + list(out_outputs[:, 1]))
        predictability = ts_IOU(list((out_labels - minv) / (maxv - minv)), list((out_outputs[:, 1] - minv) / (maxv - minv)))

        plt.title(predictability)
        plt.plot(out_labels)
        plt.plot(out_outputs[:, 1])
        plt.show()
        # exit()
        # exit()
        # print(out_outputs.shape, out_labels.shape)
        # exit()

        out_outputs = []
        out_labels = []
        num_cycles = 0
        for iv, (inputs, labels) in enumerate(test_loader):
            if iv % step_size != 0 or num_cycles >= 9:
                continue
            num_cycles += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            labels = labels.detach().cpu().numpy().squeeze()
            outputs = outputs.detach().cpu().numpy().squeeze()

            labels = labels * scale_range_y + scale_min_y
            outputs = outputs * scale_range_y + scale_min_y

            if offset_estimation:
                labels = labels + y_test_offset[iv].squeeze()
                outputs = outputs + y_test_offset[iv].squeeze()
            outputs = np.maximum(outputs, 1e-6)

            out_outputs.append(outputs)
            out_labels.append(labels)
        out_outputs = np.asarray(out_outputs)
        out_labels = np.asarray(out_labels)

        out_labels = np.reshape(out_labels, newshape=(out_labels.shape[0] * out_labels.shape[1]))
        out_outputs = np.reshape(out_outputs, newshape=(out_outputs.shape[0] * out_outputs.shape[1], out_outputs.shape[2]))

        component_ = component.replace('-pvc', '')
        if component_ not in output_pkl:
            output_pkl[component_] = {}
        output_pkl[component_][metric] = (predictability, out_labels, out_outputs)

with open('./visualization/anomaly_detection/predictions.pkl', 'wb') as f:
    pickle.dump(output_pkl, f)
